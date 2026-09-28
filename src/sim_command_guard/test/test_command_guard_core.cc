#include <gtest/gtest.h>

#include "sim_command_guard/command_guard_core.hh"

using sim_command_guard::Command;
using sim_command_guard::CommandGuardCore;

TEST(CommandGuardCore, StartupPublishesZero)
{
  CommandGuardCore guard(500000000);
  const auto outcome = guard.Step(0);
  ASSERT_TRUE(outcome.has_value());
  EXPECT_TRUE(outcome->startupZero);
  EXPECT_EQ(outcome->reason, "startup_zero");
  EXPECT_DOUBLE_EQ(outcome->command.linear, 0.0);
}

TEST(CommandGuardCore, TimeoutUsesAdvancingPhysicsTime)
{
  CommandGuardCore guard(500000000);
  guard.Step(0);
  auto applied = guard.Step(100000000, Command{0.2, 0.4, true, ""});
  ASSERT_TRUE(applied.has_value());
  EXPECT_EQ(applied->status, "forwarded");
  EXPECT_FALSE(guard.Step(100000000).has_value());
  EXPECT_FALSE(guard.Step(600000000).has_value());
  auto stopped = guard.Step(600000001);
  ASSERT_TRUE(stopped.has_value());
  EXPECT_TRUE(stopped->timeout);
  EXPECT_EQ(stopped->reason, "physics_time_timeout");
}

TEST(CommandGuardCore, TimeRewindZerosAndRejectsSameProcessCommands)
{
  CommandGuardCore guard(500000000);
  guard.Step(1000000000);
  guard.Step(1100000000, Command{0.2, 0.0, true, ""});
  auto reset = guard.Step(10);
  ASSERT_TRUE(reset.has_value());
  EXPECT_TRUE(reset->reset);
  EXPECT_TRUE(guard.ResetLatched());
  auto rejected = guard.Step(20, Command{0.2, 0.0, true, ""});
  ASSERT_TRUE(rejected.has_value());
  EXPECT_EQ(rejected->status, "rejected_zero");
  EXPECT_EQ(rejected->reason, "reset_latched_restart_required");
}

TEST(CommandGuardCore, ExplicitStopDoesNotRepeatTimeout)
{
  CommandGuardCore guard(500000000);
  guard.Step(0);
  auto stopped = guard.Step(1, Command{});
  ASSERT_TRUE(stopped.has_value());
  EXPECT_EQ(stopped->reason, "requested_zero");
  EXPECT_FALSE(guard.Step(1000000000).has_value());
}

TEST(CommandGuardCore, NonFiniteTransportInputIsRejectedWithZero)
{
  CommandGuardCore guard(500000000);
  guard.Step(0);
  auto rejected = guard.Step(1, Command{0.0, 0.0, false, "non_finite_input"});
  ASSERT_TRUE(rejected.has_value());
  EXPECT_EQ(rejected->status, "rejected_zero");
  EXPECT_EQ(rejected->reason, "non_finite_input");
}
