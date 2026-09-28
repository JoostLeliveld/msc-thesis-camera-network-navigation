#include <ignition/gazebo/System.hh>
#include <ignition/plugin/Register.hh>
#include <ignition/msgs/stringmsg.pb.h>
#include <ignition/msgs/twist.pb.h>
#include <ignition/transport/Node.hh>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "sim_command_guard/command_guard_core.hh"

namespace sim_command_guard
{
class CommandGuardSystem final : public ignition::gazebo::System,
    public ignition::gazebo::ISystemConfigure,
    public ignition::gazebo::ISystemPreUpdate
{
public:
  void Configure(const ignition::gazebo::Entity &,
      const std::shared_ptr<const sdf::Element> &sdf,
      ignition::gazebo::EntityComponentManager &,
      ignition::gazebo::EventManager &) override
  {
    if (sdf->HasElement("input_topic"))
      inputTopic_ = sdf->Get<std::string>("input_topic");
    if (sdf->HasElement("output_topic"))
      outputTopic_ = sdf->Get<std::string>("output_topic");
    if (sdf->HasElement("outcome_topic"))
      outcomeTopic_ = sdf->Get<std::string>("outcome_topic");
    double timeout = 0.5;
    if (sdf->HasElement("command_timeout"))
      timeout = sdf->Get<double>("command_timeout");
    if (!std::isfinite(timeout) || timeout <= 0.0)
      throw std::runtime_error("sim_command_guard command_timeout must be finite and positive");
    timeoutWall_ = std::chrono::duration<double>(timeout);

    core_ = std::make_unique<CommandGuardCore>(
        static_cast<std::int64_t>(timeout * 1e9));
    commandPublisher_ = node_.Advertise<ignition::msgs::Twist>(outputTopic_);
    outcomePublisher_ = node_.Advertise<ignition::msgs::StringMsg>(outcomeTopic_);
    node_.Subscribe(inputTopic_, &CommandGuardSystem::OnCommand, this);
    const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream epoch;
    epoch << std::hex << seed;
    producerEpoch_ = epoch.str();
  }

  void PreUpdate(const ignition::gazebo::UpdateInfo &info,
      ignition::gazebo::EntityComponentManager &) override
  {
    const auto simNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        info.simTime).count();
    std::optional<PendingCommand> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending.swap(pending_);
    }

    Publish(core_->Step(simNs));
    if (pending.has_value())
    {
      auto command = pending->command;
      if (std::chrono::steady_clock::now() - pending->receivedAt > timeoutWall_)
      {
        command.valid = false;
        command.rejectReason = "transport_wall_stale";
      }
      Publish(core_->Step(simNs, command));
    }
  }

private:
  void OnCommand(const ignition::msgs::Twist &message)
  {
    Command command{message.linear().x(), message.angular().z(), true, ""};
    if (!std::isfinite(command.linear) || !std::isfinite(command.angular))
      command = Command{0.0, 0.0, false, "non_finite_input"};
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ = PendingCommand{command, std::chrono::steady_clock::now()};
  }

  void Publish(const std::optional<Outcome> &outcome)
  {
    if (!outcome.has_value())
      return;
    ignition::msgs::Twist command;
    command.mutable_linear()->set_x(outcome->command.linear);
    command.mutable_angular()->set_z(outcome->command.angular);
    commandPublisher_.Publish(command);

    std::ostringstream json;
    json << std::setprecision(17)
         << "{\"schema_version\":1,\"producer_epoch\":\"" << producerEpoch_
         << "\",\"event_seq\":" << outcome->eventSeq
         << ",\"event_id\":\"" << producerEpoch_ << ':' << outcome->eventSeq
         << "\",\"command_id\":\"" << producerEpoch_ << ':' << outcome->commandSeq
         << "\",\"status\":\"" << outcome->status
         << "\",\"reason\":\"" << outcome->reason
         << "\",\"forwarded_sim_stamp_ns\":" << outcome->simTimeNs
         << ",\"requested_stamp\":null"
         << ",\"adapter_receipt_stamp\":null"
         << ",\"bridge_receipt_stamp\":null"
         << ",\"forwarded_linear\":" << outcome->command.linear
         << ",\"forwarded_angular\":" << outcome->command.angular
         << ",\"physical_application_verified\":false"
         << ",\"timeout\":" << (outcome->timeout ? "true" : "false")
         << ",\"reset\":" << (outcome->reset ? "true" : "false")
         << ",\"startup_zero\":" << (outcome->startupZero ? "true" : "false")
         << ",\"liveness_sequence\":" << outcome->eventSeq
         << '}';
    ignition::msgs::StringMsg message;
    message.set_data(json.str());
    outcomePublisher_.Publish(message);
  }

  ignition::transport::Node node_;
  ignition::transport::Node::Publisher commandPublisher_;
  ignition::transport::Node::Publisher outcomePublisher_;
  struct PendingCommand
  {
    Command command;
    std::chrono::steady_clock::time_point receivedAt;
  };
  std::mutex mutex_;
  std::optional<PendingCommand> pending_;
  std::unique_ptr<CommandGuardCore> core_;
  std::string producerEpoch_;
  std::string inputTopic_{"/model/turtlebot3/cmd_vel_guard_input"};
  std::string outputTopic_{"/model/turtlebot3/cmd_vel_applied"};
  std::string outcomeTopic_{"/model/turtlebot3/actuation_outcome"};
  std::chrono::duration<double> timeoutWall_{0.5};
};
}  // namespace sim_command_guard

IGNITION_ADD_PLUGIN(
    sim_command_guard::CommandGuardSystem,
    ignition::gazebo::System,
    sim_command_guard::CommandGuardSystem::ISystemConfigure,
    sim_command_guard::CommandGuardSystem::ISystemPreUpdate)
