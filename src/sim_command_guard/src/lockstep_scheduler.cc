#include <ignition/msgs/boolean.pb.h>
#include <ignition/msgs/world_stats.pb.h>
#include <ignition/msgs/world_control.pb.h>
#include <ignition/transport/Node.hh>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/header.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace
{
std::optional<std::string> JsonString(const std::string &json, const std::string &key)
{
  const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(json, match, expression))
    return std::nullopt;
  return match[1].str();
}
}

class LockstepScheduler final : public rclcpp::Node
{
public:
  LockstepScheduler() : Node("lockstep_scheduler")
  {
    worldName_ = declare_parameter<std::string>("world_name", "warehouse_v2");
    stepIterations_ = declare_parameter<int>("control_step_iterations", 100);
    cameraEveryControlSteps_ = declare_parameter<int>("camera_every_control_steps", 2);
    timeoutS_ = declare_parameter<double>("barrier_timeout_s", 30.0);
    startupDelayS_ = declare_parameter<double>("startup_delay_s", 2.0);
    maxControlSteps_ = declare_parameter<int>("max_control_steps", 0);
    commandQuietS_ = declare_parameter<double>("controller_quiet_s", 2.0);
    // Hold each step until the planner has handled that step's odometry. Without it a
    // planner that falls behind is not paused with the world, and its belief silently
    // ages against the simulation clock.
    waitForPlanner_ = declare_parameter<bool>("wait_for_planner", false);
    if (stepIterations_ <= 0 || cameraEveryControlSteps_ <= 0 || timeoutS_ <= 0.0)
      throw std::invalid_argument("lockstep scheduler parameters must be positive");

    detectorSub_ = create_subscription<std_msgs::msg::String>(
      "/perception/camera_batch_outcome", rclcpp::QoS(256).reliable(),
      [this](const std_msgs::msg::String &message) { OnDetector(message.data); });
    managerSub_ = create_subscription<std_msgs::msg::String>(
      "/reliability/camera_manager/batch_outcome", rclcpp::QoS(256).reliable(),
      [this](const std_msgs::msg::String &message) { OnManager(message.data); });
    odomSub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", rclcpp::QoS(100),
      [this](const nav_msgs::msg::Odometry &message) {
        latestOdomNs_ = StampNs(message.header.stamp);
        ++odomCount_;
        cv_.notify_all();
      });
    plannerOdomSub_ = create_subscription<std_msgs::msg::Header>(
      "/planner/odometry_processed", rclcpp::QoS(100),
      [this](const std_msgs::msg::Header &message) {
        std::lock_guard<std::mutex> lock(mutex_);
        plannerOdomNs_ = std::max(plannerOdomNs_, StampNs(message.stamp));
        plannerSeen_ = true;
        cv_.notify_all();
      });
    // After the logger's terminal stop the detector and manager go quiescent by
    // design; perception barriers then only stall the clock that the logger's
    // rest verification needs, so from then on the scheduler steps on odometry.
    terminalSub_ = create_subscription<std_msgs::msg::String>(
      "/experiment/terminal_stop_request",
      rclcpp::QoS(16).reliable().transient_local(),
      [this](const std_msgs::msg::String &) {
        std::lock_guard<std::mutex> lock(mutex_);
        terminalRequested_ = true;
        cv_.notify_all();
      });
    commandSub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::QoS(100),
      [this](const geometry_msgs::msg::Twist &) { ++commandCount_; cv_.notify_all(); });

    gzNode_.Subscribe("/world/" + worldName_ + "/stats", &LockstepScheduler::OnStats, this);

    worker_ = std::thread([this]() { Run(); });
  }

  ~LockstepScheduler() override
  {
    stopping_ = true;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

private:
  static std::int64_t StampNs(const builtin_interfaces::msg::Time &stamp)
  {
    return static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
  }

  void OnDetector(const std::string &json)
  {
    const auto status = JsonString(json, "status");
    if (!status) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (*status == "session_started") detectorReady_ = true;
    if (*status == "published") {
      const auto id = JsonString(json, "source_batch_id");
      if (id && *id != lastDetectorBatch_) {
        lastDetectorBatch_ = *id;
        ++detectorPublishedCount_;
      }
    }
    cv_.notify_all();
  }

  void OnManager(const std::string &json)
  {
    const auto status = JsonString(json, "status");
    const auto id = JsonString(json, "source_batch_id");
    if (!status || !id || *status != "manager_decision") return;
    std::lock_guard<std::mutex> lock(mutex_);
    lastManagerBatch_ = *id;
    cv_.notify_all();
  }

  void OnStats(const ignition::msgs::WorldStatistics &stats)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    statsSimNs_ = static_cast<std::int64_t>(stats.sim_time().sec()) * 1000000000LL +
                  stats.sim_time().nsec();
    statsPaused_ = stats.paused();
    ++statsCount_;
    cv_.notify_all();
  }

  template<typename Predicate>
  bool WaitFor(Predicate predicate, const std::string &barrier)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool ready = cv_.wait_for(
      lock, std::chrono::duration<double>(timeoutS_),
      [this, &predicate]() { return stopping_ || predicate(); });
    if (stopping_) return false;
    if (!ready) {
      // A wedged barrier leaves the world paused forever; end the run loudly
      // instead (the launch shuts the experiment down when this process exits).
      RCLCPP_FATAL(get_logger(), "lockstep barrier timed out: %s", barrier.c_str());
      std::_Exit(3);
    }
    return true;
  }

  // A bounded wait that is not an error: the controller publishes only while it
  // holds an active plan, so its silence is a state, not a wedged barrier.
  template<typename Predicate>
  bool WaitForQuietly(Predicate predicate, double timeoutS)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::duration<double>(timeoutS),
      [this, &predicate]() { return stopping_ || predicate(); }) && !stopping_;
  }

  bool Control(const ignition::msgs::WorldControl &request)
  {
    ignition::msgs::Boolean response;
    bool transportResult = false;
    const bool executed = gzNode_.Request(
      "/world/" + worldName_ + "/control", request, 5000, response, transportResult);
    return executed && transportResult && response.data();
  }

  void Run()
  {
    if (!WaitFor([this]() { return detectorReady_ && odomCount_ > 0; }, "startup")) return;
    std::this_thread::sleep_for(std::chrono::duration<double>(startupDelayS_));

    ignition::msgs::WorldControl pause;
    pause.set_pause(true);
    if (!Control(pause)) {
      RCLCPP_ERROR(get_logger(), "failed to pause /world/%s", worldName_.c_str());
      return;
    }
    // The pause lands at an arbitrary simulation time. Step once to the next camera
    // instant so that every camera frame falls at the end of a camera control step
    // (1 ms physics: one iteration is 1 ms), whatever the wall-clock start-up took.
    const std::uint64_t pausedStats = statsCount_.load();
    if (!WaitFor([this, pausedStats]() { return statsCount_ > pausedStats + 1 && statsPaused_; },
                 "paused world statistics")) return;
    std::int64_t pausedNs;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pausedNs = statsSimNs_;
    }
    const std::int64_t cameraPeriodNs =
      static_cast<std::int64_t>(stepIterations_) * cameraEveryControlSteps_ * 1000000LL;
    std::int64_t alignedNs = ((pausedNs + cameraPeriodNs - 1) / cameraPeriodNs) * cameraPeriodNs;
    // A shorter step than one control step may produce no odometry message and would
    // wedge the odometry barrier (campaign: 3 of 90 runs); go one camera period further.
    if (alignedNs - pausedNs < static_cast<std::int64_t>(stepIterations_) * 1000000LL)
      alignedNs += cameraPeriodNs;
    const std::uint64_t alignIterations =
      static_cast<std::uint64_t>((alignedNs - pausedNs + 500000LL) / 1000000LL);
    std::optional<std::string> pendingManagerBatch;
    if (alignIterations > 0) {
      const auto odomBefore = odomCount_.load();
      const auto detectorBefore = detectorPublishedCount_.load();
      ignition::msgs::WorldControl align;
      align.set_pause(true);
      align.set_multi_step(alignIterations);
      if (!Control(align)) {
        RCLCPP_ERROR(get_logger(), "Gazebo alignment step request failed");
        return;
      }
      if (!WaitFor([this, odomBefore]() { return odomCount_ > odomBefore; }, "odometry")) return;
      if (!WaitFor([this, detectorBefore]() { return detectorPublishedCount_ > detectorBefore; },
                   "detector publication at the aligned camera instant")) return;
      std::lock_guard<std::mutex> lock(mutex_);
      pendingManagerBatch = lastDetectorBatch_;
    }
    RCLCPP_INFO(get_logger(),
      "lockstep active: %d iterations/control step, camera every %d steps; paused at %.3f s, "
      "aligned to %.3f s with %lu iterations",
      stepIterations_, cameraEveryControlSteps_, pausedNs * 1e-9, alignedNs * 1e-9, alignIterations);

    const auto wallStart = std::chrono::steady_clock::now();
    std::uint64_t controlStep = 0;
    bool controllerActive = false;
    while (rclcpp::ok() && !stopping_ &&
           (maxControlSteps_ <= 0 || controlStep < static_cast<std::uint64_t>(maxControlSteps_))) {
      if (pendingManagerBatch) {
        const std::string expected = *pendingManagerBatch;
        if (!WaitFor([this, &expected]() { return terminalRequested_ || lastManagerBatch_ == expected; },
                     "manager decision for " + expected)) return;
        pendingManagerBatch.reset();
      }

      const auto odomBefore = odomCount_.load();
      const auto commandBefore = commandCount_.load();
      const auto detectorBefore = detectorPublishedCount_.load();
      ignition::msgs::WorldControl step;
      step.set_pause(true);
      step.set_multi_step(static_cast<std::uint64_t>(stepIterations_));
      if (!Control(step)) {
        RCLCPP_ERROR(get_logger(), "Gazebo step request failed at control step %lu", controlStep);
        return;
      }
      ++controlStep;
      if (!WaitFor([this, odomBefore]() { return odomCount_ > odomBefore; }, "odometry")) return;
      if (waitForPlanner_) {
        const std::int64_t stepOdomNs = latestOdomNs_.load();
        // Before the planner's first message there is no belief to age; waiting then
        // would stall startup, when the planner is still loading.
        if (!WaitFor([this, stepOdomNs]() {
              return terminalRequested_ || !plannerSeen_ || plannerOdomNs_ >= stepOdomNs;
            }, "planner odometry")) return;
      }
      // While the controller is publishing, hold the step until it has answered the
      // new state; once it falls silent (no plan yet, plan finished) stop waiting.
      if (controllerActive) {
        controllerActive = WaitForQuietly(
          [this, commandBefore]() { return commandCount_ > commandBefore; }, commandQuietS_);
        if (stopping_) return;
      } else {
        controllerActive = commandCount_ > commandBefore;
      }

      if (controlStep % static_cast<std::uint64_t>(cameraEveryControlSteps_) == 0) {
        if (!WaitFor([this, detectorBefore]() {
              return terminalRequested_ || detectorPublishedCount_ > detectorBefore;
            }, "detector publication")) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!terminalRequested_) pendingManagerBatch = lastDetectorBatch_;
      }
    }

    const double wallS = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wallStart).count();
    const double simS = static_cast<double>(controlStep * stepIterations_) * 0.001;
    RCLCPP_INFO(get_logger(),
      "lockstep summary: control_steps=%lu sim_s=%.3f wall_s=%.3f rtf=%.4f",
      controlStep, simS, wallS, wallS > 0.0 ? simS / wallS : 0.0);
  }

  std::string worldName_;
  int stepIterations_{100};
  int cameraEveryControlSteps_{2};
  double timeoutS_{30.0};
  double startupDelayS_{2.0};
  int maxControlSteps_{0};
  double commandQuietS_{2.0};
  bool waitForPlanner_{false};
  bool plannerSeen_{false};
  std::int64_t plannerOdomNs_{0};
  std::atomic<std::int64_t> latestOdomNs_{0};
  ignition::transport::Node gzNode_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  std::atomic<bool> stopping_{false};
  std::atomic<std::uint64_t> odomCount_{0};
  std::atomic<std::uint64_t> commandCount_{0};
  bool detectorReady_{false};
  bool terminalRequested_{false};
  std::atomic<std::uint64_t> detectorPublishedCount_{0};
  std::string lastDetectorBatch_;
  std::string lastManagerBatch_;
  std::int64_t statsSimNs_{0};
  bool statsPaused_{false};
  std::atomic<std::uint64_t> statsCount_{0};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr detectorSub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr managerSub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
  rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr plannerOdomSub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr commandSub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr terminalSub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LockstepScheduler>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
