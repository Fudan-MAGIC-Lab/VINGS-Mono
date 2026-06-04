import numpy as np
import math

try:
    import gtsam
except ModuleNotFoundError:
    gtsam = None

GRAVITY = 9.807

class MultiSensorState:
    def __init__(self):
        if gtsam is None:
            raise ModuleNotFoundError(
                "gtsam is required for VIO mode. Use mode=vo or install gtsam with Python>=3.9."
            )
        self.cur_t = 0.0

        """ IMU-centered states """
        self.timestamps = []            # timestamps (len == N)

        self.wTbs = []                  # poses      (len == N)
        self.vs = []                    # vels       (len == N)
        self.bs = []                    # biases     (len == N)

        self.preintegrations = []       # preintegrations (len == N)
        self.preintegrations_meas = []  # raw IMU data    (len == N)
        self.preintegration_temp = None # used for high-frequency prediction
        self.pose_temp = None           # used for high-frequency prediction

        self.gnss_valid = []            # GNSS flags (len == N)
        self.gnss_position = []         # GNSS pos   (len == N)

        self.odo_valid = []             # Odo flags  (len == N)
        self.odo_vel = []               # Odo vel    (len == N)

        self.marg_factor = None
        self.set_imu_params()
    
    def set_imu_params(self, noise = None):

        # default
        accel_noise_sigma   = 0.0
        gyro_noise_sigma    = 0.0
        accel_bias_rw_sigma = 0.0
        gyro_bias_rw_sigma  = 0.0

        if noise != None:
            accel_noise_sigma = noise[0]
            gyro_noise_sigma = noise[1]
            accel_bias_rw_sigma = noise[2]
            gyro_bias_rw_sigma = noise[3]

        measured_acc_cov = np.eye(3,3) * math.pow(accel_noise_sigma,2)
        measured_omega_cov = np.eye(3,3) * math.pow(gyro_noise_sigma,2)
        integration_error_cov = np.eye(3,3) * 0e-8
        bias_acc_cov = np.eye(3,3) * math.pow(accel_bias_rw_sigma,2)
        bias_omega_cov = np.eye(3,3) * math.pow(gyro_bias_rw_sigma,2)
        bias_acc_omega_init = np.eye(6,6) * 0e-5

        params = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
        params.setAccelerometerCovariance(measured_acc_cov)
        params.setIntegrationCovariance(integration_error_cov)
        params.setGyroscopeCovariance(measured_omega_cov)
        params.setBiasAccCovariance(bias_acc_cov)
        params.setBiasOmegaCovariance(bias_omega_cov)
        params.setBiasAccOmegaInit(bias_acc_omega_init)
        self.params = params

        params_loose = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
        params_loose.setAccelerometerCovariance(measured_acc_cov* 100)
        params_loose.setIntegrationCovariance(integration_error_cov)
        params_loose.setGyroscopeCovariance(measured_omega_cov * 100)
        params_loose.setBiasAccCovariance(bias_acc_cov)
        params_loose.setBiasOmegaCovariance(bias_omega_cov)
        params_loose.setBiasAccOmegaInit(bias_acc_omega_init)
        self.params_loose = params_loose

    def init_first_state(self,t,pos,R,vel):
        self.timestamps.append(t)
        self.wTbs.append(gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(pos)))
        self.vs.append(vel)
        self.bs.append(gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])))
        self.preintegrations.append(gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[-1]))
        self.preintegrations_meas.append([])
        self.preintegration_temp = gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[-1])
        self.gnss_valid.append(False)
        self.gnss_position.append(np.array([.0,.0,.0]))
        self.odo_valid.append(False)
        self.odo_vel.append(np.array([.0,.0,.0]))

        self.cur_t = t

    def append_imu(self, t, measuredAcc, measuredOmega):
        if t - self.cur_t > 0:
            if t-self.cur_t > 0.025: # IMU gap found, loose the IMU factor
                new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params_loose,self.bs[-1])
                for iii in range(len(self.preintegrations_meas[-1])):
                    dd = self.preintegrations_meas[-1][iii]
                    if dd[2] > 0:
                        new_preintegration.integrateMeasurement(dd[0],dd[1],dd[2])
                self.preintegrations[-1] = new_preintegration
            self.preintegrations[-1].integrateMeasurement(\
                            measuredAcc, measuredOmega, t - self.cur_t)
        # if t - self.cur_t < 0:
        #   raise Exception("may not happen")
        
        self.preintegrations_meas[-1].append([measuredAcc, measuredOmega, t - self.cur_t, t])
        # print('append_imu: ',measuredAcc,measuredOmega,t - self.cur_t,t)
        self.last_measuredAcc = measuredAcc
        self.last_measuredOmega = measuredOmega
        self.cur_t = t
    
    def append_imu_temp(self, t, measuredAcc, measuredOmega, predict_pose = False):
        if t - self.cur_t > 0:
            self.preintegration_temp.integrateMeasurement(\
                            measuredAcc, measuredOmega, t - self.cur_t)
        if predict_pose:
            prev_state = gtsam.gtsam.NavState(self.wTbs[-1],self.vs[-1])
            prev_bias = self.bs[-1]
            self.pose_temp = self.preintegration_temp.predict(prev_state, prev_bias)

    def append_img(self, t):
        self.cur_t = t
        prev_state = gtsam.gtsam.NavState(self.wTbs[-1],self.vs[-1])
        prev_bias = self.bs[-1]
        prop_state = self.preintegrations[-1].predict(prev_state, prev_bias)
        if self.preintegrations[-1].deltaTij()>1.0:
            prop_state = gtsam.gtsam.NavState(self.wTbs[-1],self.vs[-1])
            
        self.timestamps.append(t)
        self.wTbs.append(prop_state.pose())        
        self.vs.append(prop_state.velocity())
        self.bs.append(prev_bias)
        self.gnss_valid.append(False)
        self.gnss_position.append(np.array([.0,.0,.0]))
        self.odo_valid.append(False)
        self.odo_vel.append(np.array([.0,.0,.0]))

        self.preintegrations.append(\
            gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[-1]))
        self.preintegrations_meas.append([])
        self.preintegration_temp = gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[-1])
    
    # ugly implementation
    # this should be called after append_img()
    def append_gnss(self,t,pos):
        if math.fabs(self.cur_t - t) > 0.01:
            print('Skip GNSS data due to unsynchronization!!')
        else:
            self.gnss_valid[-1] = True
            self.gnss_position[-1] = pos

    def append_odo(self,t,vel):
        if math.fabs(self.cur_t - t) > 0.01:
            print('Skip ODO data due to unsynchronization!!')
        else:
            self.odo_valid[-1] = True
            self.odo_vel[-1] = vel

    def predict(self):
        prev_state = gtsam.gtsam.NavState(self.wTbs[-1],self.vs[-1])
        prev_bias = self.bs[-1]
        self.preintegrations[-1].predict(prev_state,prev_bias)