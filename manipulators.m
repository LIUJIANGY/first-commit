clc;clear;
close all;warning off;

% 建立puma560机器人右手D-H模型 
mdl_puma560;
% 系统参数设置
dt = 0.01;t = 0:dt:10;N = length(t);
xd_history = [];e_history = [];x_history = [];
% 控制器增益参数
tau = [0;0;0;0;0;0];%控制器初值
w_hat = 0.1;
K1 = 99*eye(3);K21 = 33*eye(6);
tanh = [0;0;0;0;0;0];
%预设性能参数
phi_a0 = [0;0;0];phi_b0 = [0;0;0];ka = 2;
% 瞬态性能函数式
rho01 = 0.2;rho02 = 0.2;rho03 = 0.4;
rho1 = 0.012;rho2 = 0.012;rho3 = 0.025;
alphak = 2.5;beta1k = 1;beta2k = 1;
alpha0 = [30;18;52;0;0;0];
% 预设性能函数

% 期望内力
fd1 = [0,0,0];
% 紧集范围
d1 = 0.025;d2 = 0.05;M1 = 1;
% 初始值
x0 = [0.55;0.2;0.2];x1 = [0.55;0.2;0.2];% 初始位姿
P = [0;0;0];
q = [0;-pi/6;pi/6;0;0;0];% 初始关节角
dq = zeros(6,1); % 初始速度，x_dot = y_dot = z_dot均为0
ddq = zeros(6,1);
x = [q;dq];Z = zeros(12);
GAMMA = [0;0;0;0;0;0];

for k = 1:N-1
    % 获取动力学参数
    gx = p560.gravload(q')'; % 重力项
    M = p560.inertia(q');% 惯性矩阵
    C = p560.coriolis(q',dq');% 哥氏力和离心力
    % 求雅可比矩阵
    J0 = p560.jacob0(q);
    J = J0(1:3,:);
    dx1 = J*dq;
    x1 = x1 + dx1*dt;% 运动学求轨迹x
    x_history = [x_history,x1];
    % 目标轨迹
    xd = [0.65+0.1*sin(2*pi/5*t(k));
           0.12*cos(2*pi/5*t(k))    ;
           0                         ];% 轨迹方程
    dxd = [0.04*cos((2*pi/5)*t(k))  ;
            -0.048*sin((2*pi)/5*t(k));
            0                            ];
    xd_history = [xd_history,xd];
    e = x1 - xd;
    e_history = [e_history,e];% 记录误差

    phi_a = [-beta1k*((rho01-rho1)*exp(-alphak*t(k))+rho1);
             -beta1k*((rho02-rho2)*exp(-alphak*t(k))+rho2);
             -beta1k*((rho03-rho3)*exp(-alphak*t(k))+rho3)];
    phi_b = [ beta2k*((rho01-rho1)*exp(-alphak*t(k))+rho1);
              beta2k*((rho02-rho2)*exp(-alphak*t(k))+rho2);
              beta2k*((rho03-rho3)*exp(-alphak*t(k))+rho3)];
    dphi_a = (phi_a - phi_a0)/dt; dphi_b = (phi_b - phi_b0)/dt;
    sigma = sqrt((dphi_a/phi_a)^2+(dphi_b/phi_b)^2+ka);

    % 虚拟控制输入
    alpha = pinv(J)*(dxd - K1*e +sigma*e);
    z1 = dq -alpha;
    for k2=1:6
        if z1(k2)<d1
            m(k2) = 1;
        elseif (z1(k2)>d1)&&(z1(k2)<d2)
            m(k2) = (d2^2-z1(k2)^2)/(d2^2-d1^2)*exp(((z1(k2)^2-d1^2)/(d2^2-d1^2))^2);
        else
            m(k2) = 0;
        end
        M1 = M1*m(k2);
    end
    % 切换函数设计
    Q = diag(M1);

    %预设性能
    for i=1:3
        Ea = e(i)/phi_a(i);
        Eb = e(i)/phi_b(i);
        if e(i) >= 0
            h = 1;
        else
            h = 0;
        end
        PHI = h*Eb+(1-h)*Ea;
        P(i) = [PHI.^2/((1-PHI.^2)*e(i)+eps)]';
    end

    % 求出控制器tau % 单机械臂
    G = [zeros(6);inv(M)];F = [dq; -inv(M)*C*dq-inv(M)*gx];
    temp0 = -(M*((alpha-alpha0)/dt)+C*alpha+gx);

    F_U = diag(temp0);
    for k3 = 1:6
        GAMMA(k3) = 2*sigmoid(temp0(k3)*z1(k3)/w_hat)-1;
    end
    tau = -K21*z1-J'*P-J'*fd1'-Q*norm(RBF(Z(1:6,:),M1));%-Q*F_U*GAMMA; %此为鲁棒项
    dx = F + G*tau;
    x = dx*dt + x;

    % 参数更新
    phi_a0 = phi_a;phi_b0 = phi_b;
    alpha0 = alpha;
    % %更新角度和角速度
    q = x(1:6,:);
    dq = x(7:12,:);
    % 神经网络输入信号
    Z = x;
    M1 = 1;
end

% 绘制 3D 图形
figure(1)
plot3(xd_history(1,:), xd_history(2,:), xd_history(3,:));
xlabel('X');ylabel('Y');zlabel('Z');
grid on
hold on
plot3(x_history(1,:), x_history(2,:), x_history(3,:),'r');
legend('期望轨迹','跟踪轨迹');

% 绘制误差曲线
figure(2)
subplot(3,1,1)
plot(t(1:N-1),e_history(1,:),'r');
hold on
plot(t(1:N-1),(rho01-rho1)*exp(-alphak*t(1:N-1)+rho1),'-.k');
plot(t(1:N-1),(rho1-rho01)*exp(-alphak*t(1:N-1)+rho1),'-.k');
xlabel('时间t');ylabel('Error 1(rad)');
subplot(3,1,2)
plot(t(1:N-1),e_history(2,:),'g');
hold on
plot(t(1:N-1),(rho02-rho2)*exp(-alphak*t(1:N-1)+rho2),'-.k');
plot(t(1:N-1),(rho2-rho02)*exp(-alphak*t(1:N-1)+rho2),'-.k');
xlabel('时间t');ylabel('Error 2(rad)');
subplot(3,1,3)
plot(t(1:N-1),e_history(3,:),'b');
hold on
plot(t(1:N-1),(rho03-rho3)*exp(-alphak*t(1:N-1)+rho3),'-.k');
plot(t(1:N-1),(rho3-rho03)*exp(-alphak*t(1:N-1)+rho2),'-.k');
xlabel('时间t');ylabel('Error 3(rad)');

%% RBF神经网络
function resnet = RBF(Z,M1)
    THETA1 = diag(2);
    W = zeros(6);gamma = 2;% gamma是正常数
    mu = 2;l = 200;% 200个节点，mu取2
    for j = 1:1:l
        temp = (Z(1)-1+(l-1)*W(1))^2+(Z(2)-1+(l-1)*W(2))^2+(Z(3)-1+(l-1)*W(3))^2+...
            (Z(4)-1+(l-1)*W(4))^2+(Z(5)-1+(l-1)*W(5))^2+(Z(6)-1+(l-1)*W(6))^2;
        phi(j) = exp(-(temp)/mu^2); %仿照教材设计 %网络中心节点全1
        W = THETA1*(M1*norm(phi)-gamma*W);% 神经网络权值更新率
    end
    resnet = phi; 
end