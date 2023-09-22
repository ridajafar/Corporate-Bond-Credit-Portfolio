%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Case RM 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   INPUT data are stored as:
%
%   1. ZC_curve: Table of risk-free ZC rates (cont. comp. 30/360)
%       Column #1: maturity (year frac)
%       Column #2: MID rate 
%
%   2. Q: Transition matrix for an elementary two-rating-grades Markov Process 
%       Example:
%           To ->         IG    HY     Def
%   From:           IG    74%   25%     1%
%                   HY    35%   60%     5%
%                   Def    0%    0%   100%
%   legend: IG = Investment Grade
%           HY = High Yield
%           Def= Defaulted
%
%   3. Other scalar variables: self-explainig
% 
%    OUTPUT data are stored as:
%
%   4. FV: Column vector with 1-year forward value of 2y z.c. bond,
%      each row correaponding to a rating 
%      (row #1: IG; row #2: HY; row #3: Defaulted)
%      Example: FV = [94.00; 90.00; 45.00]
%   5. A set of scalar variable with self-explanatory names
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
tic
% risk-free rates
ZC_curve = [1.0 0.05; 2.0 0.05];

% Rating transition matrix
Q = [0.5281	0.4619	0.0100;
     0.3500	0.6000	0.0500;
     0.0000	0.0000	1.0000];

%Recovery rate (\pi according to Schonbucher)
R = 0.40;                        
 
% Two year bond IG
IG_cf_schedule_2y = [0.5 0.0; 1.0 0.0; 1.5 0.0; 2.0 100.0];

% IG Z-spread (scalar, corresponding to a 2y bond)
IG_z_2y = 115/10000;

% IG 1y fwd Z-spread (scalar, corresponding to a 1y bond)
IG_z_1y = 60/10000;

% HY 1y fwd Z-spread (scalar, corresponding to a 1y bond)
HY_z_1y = 300/10000;

% Number of issuers in portfolio
N_issuers = 100;    
%N_issuers = N_issuers * 10; % Uncomment for convergence towards ASRF

% AVR correlation (Schonbucher 10.12) - uncomment for discussion
% rho=0.00;
rho=0.12;
% rho=0.24; 

% Minimum number of Monte Carlo Scenarios
N = 100000;
%N = N *10;        % 1m scenarios
%N = N * 50;       % Uncomment for convergence test (5,000,000 scenarios)


%% Q1: MtM
MtM = exp(-(ZC_curve(2,2)+IG_z_2y)*ZC_curve(2,1)) * IG_cf_schedule_2y(4,2);
disp('––– Q1: MtM of the portfolio –––')
fprintf('MtM of the portfolio: %.2f \n', MtM)
disp(' ')

%% Let's start with all bonds rated IG... (uncomment if needed)
FV = zeros(3,1); 
% ...Investment grade one year from now
FV(1) = exp(-(ZC_curve(1,2)+IG_z_1y)*ZC_curve(1,1)) * IG_cf_schedule_2y(4,2); 
% ...High Yield one year from now...
FV(2) = exp(-(ZC_curve(1,2)+HY_z_1y)*ZC_curve(1,1)) * IG_cf_schedule_2y(4,2); 
% ...Defaulted one year from now
FV(3) = exp(-ZC_curve(1,2)*ZC_curve(1,1)) * R * IG_cf_schedule_2y(4,2) ; 
% Fwd expected value
E_FV = sum(FV' .* Q(1,:));                                      
disp('––– Q2: FV of the portfolio in 1y if no downgrade –––')
fprintf('FV (all IG): %.2f \n', FV(1))
disp(' ')
disp('––– Q3: FV of the portfolio in 1y if downgrade –––')
fprintf('FV (all HY): %.2f \n', FV(2))
disp(' ')

%% Barriers and simulated P/L for a single IG issuer 
opts = optimset('Display','off');
integral_DP = @(b) quadgk(@(x) normpdf(x),-6,b) - Q(1,3);
bD = fsolve(integral_DP,0,opts);                                % Barrier to Def.
integral_down = @(b) quadgk(@(x) normpdf(x),bD,b) - Q(1,2);
bd = fsolve(integral_down,0,opts);                              % Barrier to down (HY)
bu = 1.0e6;                                                     % Barrier to up (never)
L_D = E_FV - FV(3);                 % Loss if default
L_d = E_FV - FV(2);                 % Loss if down
L_i = E_FV - FV(1);                 % Profit if unchanged
L_u = 0;                            % Profit if up (never)

% Barriers (graphical representation)
figure
x=-5:0.01:5;
plot(x,normpdf(x),'linewidth',1.5);
hold on
plot([bD bD],[0, normpdf(bD)],'r','linewidth',1.5);
plot([bd bd],[0, normpdf(bd)],'r','linewidth',1.5);

%% Monte Carlo
% Generate N standard 1-d normal variables (macro-factor)
rng(42)
Y = randn(N,1);    % Realization of the single factor

%% Idiosynchratic Montecarlo
v = rho * Y * ones(1,N_issuers) + sqrt(1-rho^2) * randn(N,N_issuers);

%% Question 4: average number of defaults
D_num = sum( v <= bD , 2 );
D_count = mean(D_num);
disp('––– Q4: Average Number of Defaults –––')
fprintf('Expected defaults: %.2f \n', D_count)
disp(' ')

%% Question 5: average number of downgrade events
d_num = sum( v > bD & v <= bd , 2 );
d_count = mean(d_num);
disp('––– Q5: Average Number of Downgrade –––')
fprintf('Expected defaults: %.2f \n', d_count)
disp(' ')

%% Questions 6 and 7 (and discussion): Credit VaR
i_num = sum( v >= bd & v < bu , 2 );
u_num = sum( v >= bu  , 2 );
L_1 = L_D*D_num;
L_2 = L_D*D_num + L_d*d_num + L_i*i_num + L_u*u_num;
VaR_1 = quantile(L_1,0.999);
disp('––– Q6: Credit VaR –––')
fprintf('Credit VaR: %.2f \n', VaR_1)
disp(' ')
VaR_2 = quantile(L_2,0.999);
disp('––– Q7: Credit VaR –––')
fprintf('Credit VaR: %.2f \n', VaR_2)
disp(' ')
toc
