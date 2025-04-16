class PID:

    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.bias = 0
        self.error_prior = 0
        self.integral_prior = 0

    def _pid(self, value_goal: float, value_actual: float, time_delta) -> float:
        error = value_goal - value_actual

        integral = self.integral_prior + error * time_delta
        derivative = (error - self.error_prior) / time_delta
        control = self.Kp * error + self.Ki * integral + self.Kd * derivative + self.bias

        self.error_prior = error
        self.integral_prior = integral

        return control

    def control(self, value_goal: float, value_actual: float, time_delta) -> float:
        return self._pid(value_goal, value_actual, time_delta)


class VelocityPID(PID):

    def control(self, value_goal: float, value_actual: float, time_delta) -> float:
        pid_distance = self._pid(value_goal, value_actual, time_delta)
        pid_veloctiy = pid_distance / time_delta
        return pid_veloctiy


class SteerPID(PID):

    def control(self, value_goal: float, value_actual: float, time_delta) -> float:
        return self._pid(value_goal, value_actual, time_delta)
