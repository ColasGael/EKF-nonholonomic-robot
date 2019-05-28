%% EKF-SLAM for non-holonomic robot

%% Framework: unknown measurement noise covariance R

%% Approach: update a prior measurement noise estimate with empirical measurements 

clear all 
close all

%% Map features
m = [[0; 0], [10; 0], [10; 10], [0; 10]];
n_f = size(m, 2);

%% Simulation parameters
% discretization
dt = 0.02; % (s)

% noise parameters
Q_x = 0.1*dt*eye(3); % process noise
Q_m = zeros(2*n_f); % no noise on features dynamics (no movement)

n_y = 2*n_f; % number of measurements
R = 0.1*eye(n_y); % measurement noise
sqrt_Q_x = sqrtm(Q_x);
sqrt_R = sqrtm(R);

% priors for measurement noise
R_over = 0.01*R; % over-confident
R_under = 100*R; % under-confident
alpha = 0.9; % moving average coefficient

% number of time steps
T = 500;

%% Initialization
% initial state
x0 = [3; 3; 0]; 

% prior on pose location
mu_x_0 = x0; % mean estimate
Sigma_x_0 = 0.1*eye(3); % covariance

% prior on features location
mu_m_0 = reshape(m, [], 1); 
Sigma_m_0 = 0.1*eye(2*n_f); % covariance

% add an offset of (1,1) to check for bias in CV
offset = 0;
mu_x_0(1:2) = mu_x_0(1:2) + offset;
Sigma_x_0 = Sigma_x_0 + offset*eye(3);
mu_m_0 = mu_m_0 + offset;
Sigma_m_0 = Sigma_m_0 + offset*eye(2*n_f); % covariance

%% Trajectory
% time history
t = linspace(0,T*dt, T+1);

% control history
u = [ones(1, T+1); sin(t)]; 

x(:,1) = x0;
% compute the state history
for i=1 : T
    % sample the process noise
    w = sqrt_Q_x * randn(3,1);
    % transition model
    [x_next, ~, ~, ~] = dynamics_model(x(:,i), m, u(:,i), dt);
    x(:,i+1) = x_next + w;  
    
    % sample the measurement noise
    v = sqrt_R * randn(size(R,1),1);    
    % measurement model
    [y_next, ~, ~] = measurement_model(x(:,i+1), m);
    y(:,i) = y_next + v;
end

%% Plots
plot_KF(x, m, u, y, mu_x_0, Sigma_x_0, mu_m_0, Sigma_m_0, dt, Q_x, Q_m, R, R_over, R_under, alpha);

%% BLANK
figure; 

%% Dynamic model
function [x_next, m_next, Jf_x, Jf_m] = dynamics_model(x, m, u, dt)
    % control
    v = u(1);
    phi = u(2);
    
    % discrete time transition model
    x_next = x + dt*[v*cos(x(3,:)); v*sin(x(3,:)); phi*ones(1,size(x,2))];
    m_next = m;
    
    % discrete time dynamics Jacobian for pose
    Jf_x = eye(3);
    Jf_x(1:2,3) = dt*v*[-sin(x(3)); cos(x(3))];
    
    % dynamics Jacobian for features
    Jf_m = eye(8);
end

%% Helper function
function angle_corr = wrap2pi(angle)
    % wrap the angle between (-pi/2, + pi/2)
    angle_corr = angle - floor(angle/pi+1/2)*pi;
end

%% Measurement model: range and bearing measurements
function [y, Jg_x, Jg_m] = measurement_model(x, m)
    % state
    p = x(1:2); % shape (2, 1)
    theta = x(3);
    m = reshape(m, 2, []);
    
    % range measurement model
    y_range = sum((p-m).^2, 1).^0.5; % shape (1, 4)
    % bearing measurement model
    y_bearing = atan((p(2) - m(2,:))./(p(1) - m(1,:))) - theta; % shape (1, 4)
    % full measurement model
    y = [y_range, y_bearing]'; % shape (8,1)
    
    % range measurement Jacobian for pose
    Jg_x_range = ([(p - m); zeros(1,4)] ./ y_range)'; % shape (4, 3)
    % range measurement Jacobian for features
    Jg_m_range = (blkdiag(m(:,1)-p, m(:,2)-p, m(:,3)-p, m(:,4)-p) ./ y_range)'; % shape (4, 8)
    
    % bearing measurement Jacobian for pose
    Jg_x_bearing = [[-(p(2) - m(2,:)); p(1) - m(1,:)]./ y_range.^2; -ones(1,4)]' ; % shape (4, 3)
    % bearing measurement Jacobian for features
    temp = [-(m(2,:)-p(2)); m(1,:)-p(1)];
    Jg_m_bearing = (blkdiag(temp(:,1), temp(:,2), temp(:,3), temp(:,4)) ./ y_range.^2)'; % shape (4, 4)
    
    % full measurement Jacobian for pose
    Jg_x = [Jg_x_range; Jg_x_bearing];
    % full measurement Jacobian for features
    Jg_m = [Jg_m_range; Jg_m_bearing];
end

%% EKF SLAM filter update without re-estimating the measurement noise
function [mu_x_new, mu_m_new, Sigma_new] = EKF_slam_update(mu_x, mu_m, Sigma, u, y, dt, Q_x, Q_m, R)
    % Predict step
    [mu_x_tilde, mu_m_tilde, A_x, A_m] = dynamics_model(mu_x, mu_m, u, dt);
    mu_tilde = [mu_x_tilde; mu_m_tilde];
    Sigma_tilde = blkdiag(A_x, A_m)*Sigma*blkdiag(A_x, A_m)' + blkdiag(Q_x, Q_m);
    
    % Update step
    [y_pred, C_x, C_m] = measurement_model(mu_x_tilde, mu_m_tilde);
    C = [C_x, C_m];
    
    % Update step: use range and bearing measurements
    mu_new = mu_tilde + Sigma_tilde*C'*inv(C*Sigma_tilde*C' + R)*(y - y_pred);
    Sigma_new = Sigma_tilde - Sigma_tilde*C'*inv(C*Sigma_tilde*C' + R)*C*Sigma_tilde;
    
    mu_x_new = mu_new(1:size(mu_x,1));
    mu_m_new = mu_new(size(mu_x,1)+1:size(mu_x,1)+size(mu_m,1));
end

%% EKF SLAM filter update with re-estimation of the measurement noise
function [mu_x_new, mu_m_new, Sigma_new, R_est] = EKF_slam_update_noise(mu_x, mu_m, Sigma, u, y, dt, Q_x, Q_m, R_est, alpha)
    % Predict step
    [mu_x_tilde, mu_m_tilde, A_x, A_m] = dynamics_model(mu_x, mu_m, u, dt);
    mu_tilde = [mu_x_tilde; mu_m_tilde];
    Sigma_tilde = blkdiag(A_x, A_m)*Sigma*blkdiag(A_x, A_m)' + blkdiag(Q_x, Q_m);
    
    % Update step
    [y_pred, C_x, C_m] = measurement_model(mu_x_tilde, mu_m_tilde);
    C = [C_x, C_m];
    
    % Moving average of empirical covariance
    R_est = alpha*R_est + (1-alpha)*(y - y_pred)*(y - y_pred)';
    
    % Update step: use range and bearing measurements
    mu_new = mu_tilde + Sigma_tilde*C'*inv(C*Sigma_tilde*C' + R_est)*(y - y_pred);
    Sigma_new = Sigma_tilde - Sigma_tilde*C'*inv(C*Sigma_tilde*C' + R_est)*C*Sigma_tilde;
    
    mu_x_new = mu_new(1:size(mu_x,1));
    mu_m_new = mu_new(size(mu_x,1)+1:size(mu_x,1)+size(mu_m,1));
end

%% Plotting function for error ellipse
function plot_EE(mu, sigma)
    % parameter
    P = 0.95; 
    
    % define the ellipse radius
    eps = (1-P)/(2*pi*det(sigma)^0.5);
    eps_prime = eps*det(sigma)^0.5;
    radius = (-2*log(2*pi*eps_prime))^0.5;
    
    % define the ellipse contour
    theta = 0 : 0.01 : 2*pi;
    u = radius*[cos(theta); sin(theta)];
        
    % compute the square root
    sqrt_sigma = sqrtm(sigma);
    
    % reparametrization
    x = sqrt_sigma*u + mu;
    
    scatter(mu(1), mu(2), 'x', 'green');
    plot(x(1,:), x(2,:), 'LineWidth', 1, 'color', 'red');
end

%% Kalman filter plot
function plot_KF(x, m, u, y, mu_x_0, Sigma_x_0, mu_m_0, Sigma_m_0, dt, Q_x, Q_m, R, R_over, R_under, alpha)
    % number of time steps
    T = size(x,2)-1;
    % time history
    t = linspace(dt,T*dt, T);

    % initialization
    mu_x = mu_x_0;
    mu_m = mu_m_0;
    Sigma = blkdiag(Sigma_x_0, Sigma_m_0);
    mu_x_correct = mu_x_0;
    mu_m_correct = mu_m_0;
    Sigma_correct = Sigma;
    R_correct = R;
    mu_x_over = mu_x_0;
    mu_m_over = mu_m_0;
    Sigma_over = Sigma;
    mu_x_under = mu_x_0;
    mu_m_under = mu_m_0;
    Sigma_under = Sigma;
    
    % Reference plot: classic EKF SLAM with exact measurement noise covariance
    figure(1); hold on; 
    % plot the trajectory used 
    plot(x(1,1:end-1), x(2,1:end-1));
    % plot the map features
    scatter(m(1,:), m(2,:), '+', 'r'); hold off

    % Test plots: EKF SLAM with re-estimation of the measurement noise from different priors
        % correct measurement noise covariance prior
    figure(2); hold on;
    % plot the trajectory used 
    plot(x(1,1:end-1), x(2,1:end-1));
    % plot the map features
    scatter(m(1,:), m(2,:), '+', 'r'); hold off
        % overconfident measurement noise covariance prior
    figure(3); hold on; 
    % plot the trajectory used 
    plot(x(1,1:end-1), x(2,1:end-1)); 
    % plot the map features
    scatter(m(1,:), m(2,:), '+', 'r'); hold off
        % underconfident measurement noise covariance prior
    figure(4); hold on; 
    % plot the trajectory used 
    plot(x(1,1:end-1), x(2,1:end-1)); 
    % plot the map features
    scatter(m(1,:), m(2,:), '+', 'r'); hold off
    
    % compute the belief history
    for i=1 : T       
        % Reference plot: classic EKF SLAM with exact measurement noise covariance
        [mu_x, mu_m, Sigma] = EKF_slam_update(mu_x, mu_m, Sigma, u(:,i), y(:,i), dt, Q_x, Q_m, R);
        mus_x(:,i) = mu_x;
        if mod(i, floor(T/10)) == 0
            % plot the error ellipse on the robot position
            figure(1); hold on;
            plot_EE(mu_x(1:2), Sigma(1:2,1:2)); hold off
        end
        
        % Test plots: EKF SLAM with re-estimation of the measurement noise from different priors
            % correct measurement noise covariance
        [mu_x_correct, mu_m_correct, Sigma_correct, R_correct] = EKF_slam_update_noise(mu_x_correct, mu_m_correct, Sigma_correct, u(:,i), y(:,i), dt, Q_x, Q_m, R_correct, alpha);
        mus_x_correct(:,i) = mu_x_correct;
        if mod(i, floor(T/10)) == 0
            % plot the error ellipse on the robot position
            figure(2); hold on;
            plot_EE(mu_x_correct(1:2), Sigma_correct(1:2,1:2)); hold off
        end
            % overconfident measurement noise covariance
        [mu_x_over, mu_m_over, Sigma_over, R_over] = EKF_slam_update_noise(mu_x_over, mu_m_over, Sigma_over, u(:,i), y(:,i), dt, Q_x, Q_m, R_over, alpha);
        mus_x_over(:,i) = mu_x_over;
        if mod(i, floor(T/10)) == 0
            % plot the error ellipse on the robot position
            figure(3); hold on;
            plot_EE(mu_x_over(1:2), Sigma_over(1:2,1:2)); hold off
        end
            % underconfident measurement noise covariance
        [mu_x_under, mu_m_under, Sigma_under, R_under] = EKF_slam_update_noise(mu_x_under, mu_m_under, Sigma_under, u(:,i), y(:,i), dt, Q_x, Q_m, R_under, alpha);
        mus_x_under(:,i) = mu_x_under;
        if mod(i, floor(T/10)) == 0
            % plot the error ellipse on the robot position
            figure(4); hold on;
            plot_EE(mu_x_under(1:2), Sigma_under(1:2,1:2)); hold off
        end
        
    end
    
    % Reference plot: classic EKF SLAM with exact measurement noise covariance
    figure(1); hold on;
    plot(mus_x(1,:), mus_x(2,:), 'g')
    legend(["trajectory p", "features position", "position mean", "error ellipse"]);
    title("EKF SLAM with correct measurement noise covariance R"); hold off
    norm(x(:,2:end) - mus_x)
    norm(R- R)
    
    % Test plots: EKF SLAM with re-estimation of the measurement noise from different priors
    figure(2); hold on;
    plot(mus_x_correct(1,:), mus_x_correct(2,:), 'g')
    legend(["trajectory p", "features position", "position mean", "error ellipse"]);
    title("EKF SLAM with estimation of the measurement noise covariance R from a correct prior"); hold off
    norm(x(:,2:end) - mus_x_correct)
    norm(R- R_correct)
    figure(3); hold on;
    plot(mus_x_over(1,:), mus_x_over(2,:), 'g')
    legend(["trajectory p", "features position", "position mean", "error ellipse"]);
    title("EKF SLAM with estimation of the measurement noise covariance R from an overconfident prior"); hold off
    norm(x(:,2:end) - mus_x_over)
    norm(R-R_over)
    figure(4); hold on;
    plot(mus_x_under(1,:), mus_x_under(2,:), 'g')
    legend(["trajectory p", "features position", "position mean", "error ellipse"]);
    title("EKF SLAM with estimation of the measurement noise covariance R from an underconfident prior"); hold off
    norm(x(:,2:end) - mus_x_under)
    norm(R-R_under)
end
