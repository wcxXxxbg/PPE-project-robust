load('demand22-10-4_25hospital21dept6kits.mat');
yalmip('clear');
% 约束会因为...换行出问题
%Ver3: 可求解，结果与一阶段鲁棒相同，但x,s方案不同，其中平稳分配和第一天分配大多数库存取决于是否将k=0的情况纳入主问题。
%2023/9/3 adaptive robust model,\mu is variable for big-m, \lambda is dual
%variable
%% input
d_bar = dailydemand;
d_hat = divdailydemand2;
I = size(a,1);
H = size(d_bar,1);
K = size(d_bar,3);
J = size(d_bar,2);
T = size(d_bar,4);
R = T;
current_state = 1;
horizon = 3;
Capacitybaseline = [182027,100393,262093,189639,212562,200149]';
% Capacity = [0.8.*Capacitybaseline(1),0.9.*Capacitybaseline(2),1.1.*Capacitybaseline(3),1.3.*Capacitybaseline(4),...
%     1.2.*Capacitybaseline(5),0.9.*Capacitybaseline(6)];
Capacity = 1*Capacitybaseline;
c = [1.5,4,6,6,2,2]'; %cost
cost = repmat(c,[1 H T]);
meanpen = [30.5 28 11 3 17.5 25.5]'; %penalty term
penalty = repmat(p,[1 1 T]);
max_iter = 15;
count = 25;
totalcost = ones(1,count);
s_cost = ones(1,count);
ctrecord_x = ones(I,H,T,count);
ctrecord_s = ones(J,K,T,count);
ctrecord_y = ones(H,J,K,T,count);
x_record = ones(I,H,T,T-horizon);
s_record = ones(J,K,T,T-horizon);
y_record = ones(H,J,K,T,T-horizon);
inventory_record = ones(I,H,T,T-horizon);
lam_record = ones(J,K,T,T-horizon);
muu_record = ones(H,J,K,T,T-horizon);
ssp_record = ones(J,K,T,T-horizon);
ysp_record = ones(H,J,K,T,T-horizon);
 % for ct = 1:count
ct = 15;
%% variables
vx = ones(I,H,T,max_iter);
vveta = ones(1,max_iter);
vs = zeros(J,K,T,max_iter);
vmuu = zeros(H,J,K,T,max_iter);
vlam = zeros(J,K,T,max_iter);
vs_sum = zeros(1,max_iter);
gamma = ct*ones(J,K,T);
% for current_state = 1:T-horizon
    iter = 1;
    x = sdpvar(I,H,T);
    s = sdpvar(J,K,T);
    s_sp = sdpvar(J,K,T);
    inventory = sdpvar(I,H,T);
    inventory_sp = sdpvar(I,H,T);
    y = sdpvar(H,J,K,T);
    y_sp = sdpvar(H,J,K,T);
    muu_1 = sdpvar(I,H,J,K,T,R);
    muu_2 = sdpvar(H,J,K,T);
    muu_3 = sdpvar(H,J,K,T,R);
    muu_4 = sdpvar(I,H,J,K,T+1,R);
    lam_1 = sdpvar(I,H,T);
    lam_2 = sdpvar(J,K,T);
    lam_3 = sdpvar(I,H,T+1);
    lam_4 = sdpvar(I,H,T);
    x_sum = sdpvar(1);
    s_sum = sdpvar(1);
    s_sp_sum = sdpvar(1);
    eta = sdpvar(1);
    alpha = sdpvar(H,J,K,T);
    beta = sdpvar(H,J,K,T,R);
    delta = binvar(H,J,K,T);
    theta = sdpvar(J,K,T);
    Theta = 4*ones(J,K);
    M_1 = 10000;
    M_2 = 10000;
    M_3 = 10000;
    M_4 = 10000;
%% MP
%y and x
C_mp = [];
% for t = current_state:current_state+horizon
for t = 1:T
    for h = 1:H
        for i = 1:I
            C_mp = [C_mp,x(i,h,t) >= sum(a(i,:)*squeeze(y(h,:,:,t))') ];
        end
    end
end
% end

% C_mp = [C_mp,d_bar(:,:,:,current_state:current_state+horizon)<=y(:,:,:,current_state:current_state+horizon),y(:,:,:,current_state:current_state+horizon)<=d_bar(:,:,:,current_state:current_state+horizon)+d_hat(:,:,:,current_state:current_state+horizon)];
C_mp = [C_mp,d_bar<=y,y<=d_bar+d_hat];

%Capacity
for i = 1:I
    C_mp = [C_mp,sum(sum(x(i,:,:))) <= Capacity(i)];
end

C_mp = [C_mp,x>=0,s>=0,x_sum == sum(sum(sum(x.*cost))),s_sum == sum(sum(sum((s.*penalty))))];

% plane cut and objective function
C_mp = [C_mp,eta>=s_sum];
obj_mp = x_sum + eta;

%% CCG
LB = -inf;
UB = inf;
while UB-LB>=1

    %solve MP
    ops_mp = sdpsettings('solver','gurobi','verbose',1);
    result_mp = optimize(C_mp,obj_mp,ops_mp);
    vx(:,:,:,iter) = value(x);
    vs_sum = value(s_sum);
    vs(:,:,:,iter) = value(s);
    vveta(iter) = value(eta);
    LB = value(obj_mp);
    vy(:,:,:,:,iter) = value(y);
%% SP
% C_sp = [d_bar(:,:,:,current_state:current_state+horizon) <= y_sp(:,:,:,current_state:current_state+horizon) <= d_bar(:,:,:,current_state:current_state+horizon) + d_hat(:,:,:,current_state:current_state+horizon),muu >= 0,lam >= 0];
% Dual SP 37
for i = 1:I
    for k = 1:K
        for t = 1:T
            for j = 1:J
            C_sp = [sum(a(i,k).*lam_1(i,:,t)) + lam_2(j,k,t) + sum(a(i,k).*lam_3(i,:,t+1)) ==0];
            end
        end
    end
end
% Dual SP 38, BigM muu2 muu3
for h = 1:H
    for j = 1:J
        for k = 1:K
            for t = 1:T
                for r = 1:R
                    C_sp = [C_sp, sum(d_bar(h,j,k,r).*(a(:,k)'*lam_1(:,h,t))) + sum(d_hat(h,j,k,r).*(a(:,k)'*muu_1(:,h,j,k,t,r))) + d_bar(h,j,k,t)*lam_2(j,k,t) + d_hat(h,j,k,t)*muu_2(h,j,k,t) + sum(d_bar(h,j,k,r).*(a(:,k)'*lam_3(:,h,t+1))+ sum(d_hat(h,j,k,r).*(a(:,k)'*muu_4(:,h,j,k,t+1,r))))==0]; 
                    C_sp = [C_sp,muu_2(h,j,k,t) <= lam_2(j,k,t), muu_2(h,j,k,t) <= M_2*delta(h,j,k,t),muu_2(h,j,k,t)>= lam_2(j,k,t) - M_2*(1 - delta(h,j,k,t))];
                    C_sp = [C_sp, muu_3(h,j,k,t,r) <= lam_2(j,k,t), muu_3(h,j,k,t,r) <= M_3*delta(h,j,k,r), muu_3(h,j,k,t,r)>= lam_2(j,k,t) - M_3*(1 - delta(h,j,k,r))];
                end
            end
        end
    end
end
% Transform the sum terms into matrix operations
A = repmat(reshape(a, [I, 1, 1, K, 1, 1]), [1, H, J, 1, T, R]);
L1 = repmat(reshape(lam_1, [I, H, 1, 1, T, 1]), [1, 1, J, K, 1, R]);
M1 = repmat(reshape(muu_1, [I, H, J, K, T, 1]), [1, 1, 1, 1, 1, R]);

term1 = sum(A .* d_bar .* L1, 1);
term2 = sum(A .* d_hat .* M1, 1);

% Reshape lam_2, muu_2, d_bar, and d_hat for further operations
L2 = reshape(lam_2, [1, J, K, T, 1]);
M2 = reshape(muu_2, [H, J, K, T, 1]);
DBar = reshape(d_bar, [H, J, K, 1, T]);
DHat = reshape(d_hat, [H, J, K, 1, T]);

term3 = DBar .* L2;
term4 = DHat .* M2;

L3 = repmat(reshape(lam_3, [I, H, 1, 1, T+1, 1]), [1, 1, J, K, 1, R]);
M4 = repmat(reshape(muu_4, [I, H, J, K, 1, R]), [1, 1, 1, 1, T+1, 1]);

term5 = sum(A .* d_bar .* L3, 1);
term6 = sum(A .* d_hat .* M4, 1);

% Combine all terms
constraints = term1 + term2 + term3 + term4 + term5 + term6;

% Add constraints to C_sp

% Other constraints
C_sp = [C_sp, muu_2(:) <= lam_2(:), muu_2(:) <= M_2*delta(:), muu_2(:) >= lam_2(:) - M_2*(1 - delta(:))];
C_sp = [C_sp, muu_3(:) <= lam_2(:), muu_3(:) <= M_3*delta(:), muu_3(:) >= lam_2(:) - M_3*(1 - delta(:))];
%Dual SP 39
for i = 1:I
    for h = 1:H
        for t = 1
            C_sp = [C_sp, -lam_1(i,h,t) + lam_3(i,h,t) - lam_3(i,h,t+1) + lam_4(i,h,t) <= 0];
        end
    end
end
%Dual SP 40
for t = 2:T
    C_sp = [C_sp, -lam_1(:,:,t) + lam_3(:,:,t) - lam_3(:,:,t+1) <= 0];
end

%Dual SP 41
for t = 1:T
    C_sp = [C_sp,lam_2(:,:,t) <= p];
end
%Dual SP 42
C_sp = [C_sp, lam_1>=0];

% BigM muu1 muu4
for i = 1:I
    for h = 1:H
        for j = 1:J
            for k = 1:K
                for t = 1:T
                    for r = 1:R
                        C_sp = [C_sp, muu_1(i,h,j,k,t,r) <= lam_1(i,h,t),muu_1(i,h,j,k,t,r)<= M_1*delta(h,j,k,r), muu_1(i,h,j,k,t,r)>= lam_1(i,h,t) - M_1*(1-delta(h,j,k,r))];
                        C_sp = [C_sp, muu_4(i,h,j,k,t,r) <= lam_3(i,h,t+1), muu_4(i,h,j,k,t,r)<= M_4*delta(h,j,k,r), muu_4(i,h,j,k,t,r)>= lam_3(i,h,t+1) - M_4*(1 - delta(h,j,k,r))];
                    end
                end
            end
        end
    end
end
%Unicerntain set
c_sp = [C_sp, sum(delta,4) <= theta.*gamma, sum(theta,3) <= Theta];

x_shifted = cat(3, zeros(I, H, 1,1), vx(:,:,1:end-1,iter)); 
% objective function
obj_sp = sum(sum(sum(lam_1.*vx(:,:,:,iter)))) + sum(sum(sum(sum(lam_2.*d)))) + sum(sum(sum(lam_3.*x_shifted)));
ops_sp = sdpsettings('solver','gurobi','verbose',1,'gurobi.DualReductions',0);
result_sp = optimize(C_sp,obj_sp,ops_sp);
vy_sp = value(y_sp);
vinventory_sp = value(inventory_sp);
vmuu(:,:,:,:,iter+1) = value(muu);
vlam(:,:,:,iter+1) = value(lam);
vs_sp = value(s_sp);
UB = min(value(obj_mp) - value(eta) + value(obj_sp),UB);

%% add MP constraint
%y and x
% C_mp = [];
C_mp = [C_mp,inventory(:,:,1) == 0,0 <= inventory,x>=0,s >= 0,d_bar<= y<= d_bar+ d_hat];
for t = current_state:current_state+horizon
    for h = 1:H
        for i = 1:I
%             C_mp = [C_mp,sum(a(i,:)*squeeze(y(h,:,:,t))') <= x(i,h,t) + squeeze(inventory(i,h,t))];
            C_mp = [C_mp,sum(a(i,:)*squeeze(y(h,:,:,t))') <= x(i,h,t) ];
        end
    end
end


% shortage
for k = 1:K
    for t = current_state:current_state+horizon
        C_mp= [C_mp,s(:,k,t)' == sum(d_bar(:,:,k,t),1) + sum(vmuu(:,:,k,t,iter+1),1) + (vlam(:,k,t,iter+1).*gamma(:,k,t))'- sum(y(:,:,k,t),1)];
    end
end

% flow conservation
% for t = current_state+1:current_state+horizon
%     for h = 1:H
%         for i = 1:I
%             C_mp = [C_mp,inventory(i,h,t-1) + x(i,h,t-1) - sum(a(i,:)*squeeze(y(h,:,:,t-1))') == inventory(i,h,t)];
%         end
%     end
% end


for i = 1:I
    C_mp = [C_mp,sum(sum(x(i,:,:))) <= Capacity(i)];
end
C_mp = [C_mp,x_sum == sum(sum(sum(x.*cost))),s_sum == sum(sum(sum((s.*penalty))))];
C_mp = [C_mp,eta>=s_sum];
obj_mp = x_sum + eta;
% if iter>= 10
%     break
% end
display(['UB: ',num2str(UB),' LB: ',num2str(LB),'iter: ',num2str(iter)]);
iter = iter+1;
% vmuu2 = vmuu(:,:,:,:,2);
% vmuu3 = vmuu(:,:,:,:,3);
end
x_record(:,:,:,current_state) = vx(:,:,:,iter-1);
y_record(:,:,:,:,current_state) = vy(:,:,:,:,iter-1);
s_record(:,:,:,current_state) = vs(:,:,:,iter-1);
lam_record(:,:,:,current_state) = vlam(:,:,:,iter);
muu_record(:,:,:,:,current_state) = vmuu(:,:,:,:,iter);
vinventory = value(inventory);
inventory_record(:,:,:,current_state) = vinventory(:,:,:);
ssp_record(:,:,:,current_state) = value(s_sp);
ysp_record(:,:,:,:,current_state) = value(y_sp);
display(['current_state:',num2str(current_state)]);
% end
totalcost(ct) = value(obj_mp);
s_cost(ct) = value(s_sum);
ctrecord_x(:,:,:,ct) = value(x);
ctrecord_s(:,:,:,ct) = value(s);
ctrecord_y(:,:,:,:,ct) = value(y);
ct
% end
