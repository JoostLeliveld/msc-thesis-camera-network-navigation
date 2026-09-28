#ifndef SIM_COMMAND_GUARD__COMMAND_GUARD_CORE_HH_
#define SIM_COMMAND_GUARD__COMMAND_GUARD_CORE_HH_

#include <cstdint>
#include <optional>
#include <string>

namespace sim_command_guard
{
struct Command
{
  double linear{0.0};
  double angular{0.0};
  bool valid{true};
  std::string rejectReason;
};

struct Outcome
{
  Command command{};
  std::string status;
  std::string reason;
  std::uint64_t eventSeq{0};
  std::uint64_t commandSeq{0};
  std::int64_t simTimeNs{0};
  bool timeout{false};
  bool reset{false};
  bool startupZero{false};
};

class CommandGuardCore
{
public:
  explicit CommandGuardCore(std::int64_t timeoutNs) : timeoutNs_(timeoutNs) {}

  std::optional<Outcome> Step(std::int64_t simTimeNs,
      const std::optional<Command> &incoming = std::nullopt)
  {
    if (!initialized_)
    {
      initialized_ = true;
      lastSimTimeNs_ = simTimeNs;
      return MakeZero("startup_zero", false, false, true, simTimeNs);
    }

    if (simTimeNs < lastSimTimeNs_)
    {
      lastSimTimeNs_ = simTimeNs;
      lastCommandTimeNs_.reset();
      stopped_ = true;
      resetLatched_ = true;
      return MakeZero("simulation_time_rewind", false, true, false, simTimeNs);
    }
    lastSimTimeNs_ = simTimeNs;

    if (resetLatched_)
    {
      if (incoming.has_value())
      {
        ++commandSeq_;
        auto rejected = MakeZero("reset_latched_restart_required", false, true,
            false, simTimeNs);
        rejected.status = "rejected_zero";
        return rejected;
      }
      return std::nullopt;
    }

    if (incoming.has_value())
    {
      ++commandSeq_;
      lastCommandTimeNs_ = simTimeNs;
      if (!incoming->valid)
      {
        stopped_ = true;
        auto rejected = MakeZero(
            incoming->rejectReason.empty() ? "invalid_input" : incoming->rejectReason,
            false, false, false, simTimeNs);
        rejected.status = "rejected_zero";
        return rejected;
      }
      stopped_ = IsZero(*incoming);
      return Make(*incoming, "forwarded", stopped_ ? "requested_zero" : "command",
          simTimeNs);
    }

    if (!stopped_ && lastCommandTimeNs_.has_value() &&
        simTimeNs - *lastCommandTimeNs_ > timeoutNs_)
    {
      stopped_ = true;
      return MakeZero("physics_time_timeout", true, false, false, simTimeNs);
    }
    return std::nullopt;
  }

  bool ResetLatched() const {return resetLatched_;}

private:
  static bool IsZero(const Command &command)
  {
    return command.linear == 0.0 && command.angular == 0.0;
  }

  Outcome Make(const Command &command, const std::string &status,
      const std::string &reason, std::int64_t simTimeNs)
  {
    return Outcome{command, status, reason, ++eventSeq_, commandSeq_, simTimeNs};
  }

  Outcome MakeZero(const std::string &reason, bool timeout, bool reset,
      bool startup, std::int64_t simTimeNs)
  {
    auto outcome = Make(Command{}, "forwarded_zero", reason, simTimeNs);
    outcome.timeout = timeout;
    outcome.reset = reset;
    outcome.startupZero = startup;
    return outcome;
  }

  std::int64_t timeoutNs_;
  bool initialized_{false};
  bool stopped_{true};
  bool resetLatched_{false};
  std::int64_t lastSimTimeNs_{0};
  std::optional<std::int64_t> lastCommandTimeNs_;
  std::uint64_t eventSeq_{0};
  std::uint64_t commandSeq_{0};
};
}  // namespace sim_command_guard

#endif  // SIM_COMMAND_GUARD__COMMAND_GUARD_CORE_HH_
