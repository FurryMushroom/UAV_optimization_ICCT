import torch
import torch.autograd
from torch.distributions import Normal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from _datetime import datetime as dt
import time

start_time = time.time()

Nt = 5  # Nt denotes the number of isotropic antenna elements
β0 = 1e-3  # β0 represents the channel power gain at a reference distance of unit meter
square_σ = 1e-14  # square_σ denotes the power of AWGN at the receiver side,
theta = np.pi / 12  # the angle in which antenna can gain radiation
NUM_OF_UAVS = 4
NUM_OF_TARGETS = 3
SAFE_REGION_OF_UAVS = 1600
# TRANSMIT_POWER_OF_TARGETS=torch.FloatTensor(NUM_OF_TARGETS,1)
TRANSMIT_POWER_OF_TARGETS = torch.tensor([5e-10, 2e-10, 2e-10])
TRANSMIT_POWER_OF_INTEFERERS = .015
M = Normal(torch.tensor(0.), torch.tensor(1.))
MIN_DISTANCE_BETWEEN_UAV_AND_TARGET = torch.tensor(500.)
MIN_DISTANCE_BETWEEN_UAVS = torch.tensor(50.)
q_xy = torch.FloatTensor(NUM_OF_UAVS, 2)  # position of UAVs
H = 600.  # height of UAVs
qt = torch.FloatTensor(NUM_OF_TARGETS, 3)  # position of targets
q_xy = torch.tensor([
    # [1595, 2451],
  [1592.6812, 1118.4830],
      [1478.9442, 1750.4237],
      [1600.0000, 2614.4233],
      [1451.8496, 1697.8922]
    # [1320,1850]
])

# q_xy = torch.tensor([
#     [300., 600],
#                      [200.,500]])
M1 = torch.tensor([[1., 0, 0], [0, 1, 0]])
M2 = torch.tensor([0., 0., H]).repeat(NUM_OF_UAVS, 1)
# qt = torch.tensor([[2700., 2800, 500],
#                    # caution:location of UAVs at any dimension cannot be the same,will cause inf value and lead to error
#                    [2200, 1800, 500],
#                    [3200, 1900, 500]
#                    ])
qt = torch.tensor([[2900., 2600, 500],
                   # caution:location of UAVs at any dimension cannot be the same,will cause inf value and lead to error
                   [3700, 1000, 500],
                   [1900, 1500, 500]
                   ])
azimuth = torch.ones(NUM_OF_UAVS, dtype=torch.float32)  # denotes antenna azimuth of UAVs
azimuth = torch.tensor([0.8494, -0.3277, -0.0111, -0.3029])


def q_(q_xy_temp):
    q_temp = torch.matmul(q_xy_temp, M1) + M2
    return q_temp


def pos_relative():
    q_unfolded = q.unfold(1, 3, 1)  # or the unfold function will be executed NUM_OF_TARGETS times,it seems
    position_relative = q_unfolded.repeat(1, NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1)
    # unfold add one dimension,and repeat NUM_OF_TARGETS times at the new dimension,then flatten to 2 dims
    return position_relative


def relative_hori_angle():
    position_relative = pos_relative()
    relative_horizontal_angle = torch.arctan(position_relative[:, 1] / position_relative[:, 0])
    return relative_horizontal_angle


def angle_range():
    relative_angle = relative_hori_angle().reshape((NUM_OF_UAVS, NUM_OF_TARGETS))
    max_relative_angle = torch.max(relative_angle, dim=1)
    min_relative_angle = torch.min(relative_angle, dim=1)
    return min_relative_angle, max_relative_angle


def sigma_gamma(q_xy_temp, azimuth_temp):
    # at first this was partitioned into several functions,but leads to repetitive calculation
    # position_relative
    q_temp = q_(q_xy_temp)
    q_unfolded = q_temp.unfold(1, 3, 1)  # or the unfold function will be executed NUM_OF_TARGETS times,it seems
    position_relative = q_unfolded.repeat(1, NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1)
    # unfold add one dimension,and repeat NUM_OF_TARGETS times at the new dimension,then flatten to 2 dims
    # relative_horizontal_angle
    relative_horizontal_angle = torch.arctan(position_relative[:, 1] / position_relative[:, 0])
    distance_between_UAVs_and_targets = torch.sqrt(
        position_relative[:, 0].pow(2) + position_relative[:, 1].pow(2) + position_relative[:, 2].pow(2))
    distance_between_UAVs_and_targets = torch.reshape(distance_between_UAVs_and_targets,
                                                      (NUM_OF_TARGETS * NUM_OF_UAVS, 1))
    directional_signal_gain = torch.ones((NUM_OF_TARGETS * NUM_OF_UAVS, 1), dtype=torch.float32)
    for i in range(0, NUM_OF_TARGETS * NUM_OF_UAVS):
        if abs(relative_horizontal_angle[i] - azimuth_temp[
            i // NUM_OF_TARGETS]) < 0.8 * theta:  # caution:here the relative pitch angle was not taken into account
            directional_signal_gain[i] *= torch.exp(
                -((relative_horizontal_angle[i] - azimuth_temp[i // NUM_OF_TARGETS]).pow(2)) / 2 / theta ** 2
            )
        elif abs(relative_horizontal_angle[i] - azimuth_temp[i // NUM_OF_TARGETS]) < theta:
            directional_signal_gain[i] *= torch.exp(-torch.tensor(0.8 ** 2 / 2)) * 5 * (
                    1 - 1 * abs(relative_horizontal_angle[i] - azimuth_temp[i // NUM_OF_TARGETS]) / theta)
        else:
            directional_signal_gain[i] *= 1e-9 * azimuth_temp[i // NUM_OF_TARGETS]  # render requires_grad=True
    powers = β0 * Nt * TRANSMIT_POWER_OF_INTEFERERS * directional_signal_gain * distance_between_UAVs_and_targets.pow(
        - 2)
    powers = torch.reshape(powers, (NUM_OF_UAVS, NUM_OF_TARGETS))
    powers_on_each_target = torch.cumsum(powers, dim=0)[NUM_OF_UAVS - 1]
    gamma = TRANSMIT_POWER_OF_TARGETS / (powers_on_each_target + square_σ)
    sigma_gamma = torch.cumsum(gamma, dim=0)[NUM_OF_TARGETS - 1]
    return sigma_gamma


def plot_fig():
    fig = plt.figure()  # 定义画布
    x_major_locator = MultipleLocator(500)  # 把x轴的刻度间隔设置，并存在变量里
    y_major_locator = MultipleLocator(500)
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(0, 4500)  # 把x轴的刻度范围设置，不满一个刻度间隔数字不会显示出来，但是能看到一点空白
    plt.ylim(0, 4500)
    LENGTH = 4000
    x = np.array([i for i in range(0, 1600)])
    plt.fill_between(x, 0, 4500, facecolor='green', alpha=0.2, label='Deployable Area')

    def plot_edge_ray(start_x, start_y, angle):  # plot a pseudo ray
        for i in range(0, start_x.size):
            end_x = start_x[i] + LENGTH * np.cos(angle[i])
            end_y = start_y[i] + LENGTH * np.sin(angle[i])
            ray_x = np.append(start_x[i], end_x)
            ray_y = np.append(start_y[i], end_y)
            plt.plot(ray_x, ray_y, "g:", linewidth=.5)

    def plot_axis_ray(start_x, start_y, angle):  # plot a pseudo ray
        for i in range(0, start_x.size):
            print(start_x[i])
            end_x = start_x[i] + LENGTH * np.cos(angle[i])
            end_y = start_y[i] + LENGTH * np.sin(angle[i])
            ray_x = np.append(start_x[i], end_x)
            ray_y = np.append(start_y[i], end_y)
            plt.plot(ray_x, ray_y, color="dodgerblue", linestyle='--', linewidth=.6)

    q_plot = q_xy.detach().numpy()
    azi = azimuth.detach().numpy()
    plt.scatter(q_plot[:, 0], q_plot[:, 1], c="b", marker='*', label='UAV')
    plt.scatter(qt[:, 0], qt[:, 1], c="r", marker='x', label='Target')
    plot_axis_ray(q_plot[:, 0], q_plot[:, 1], azi)
    plot_edge_ray(q_plot[:, 0], q_plot[:, 1], azi - theta)
    plot_edge_ray(q_plot[:, 0], q_plot[:, 1], azi + theta)
    has_labeled_circle = False;
    for i in range(0, NUM_OF_TARGETS):
        if not has_labeled_circle:
            draw_circle = plt.Circle((qt[i][0], qt[i][1]), MIN_DISTANCE_BETWEEN_UAV_AND_TARGET, color='lightsalmon',
                                     fill=False, ls='--', linewidth=0.4, label='Minimum UAV-target distance')
            has_labeled_circle=True;
        else:
            draw_circle = plt.Circle((qt[i][0], qt[i][1]), MIN_DISTANCE_BETWEEN_UAV_AND_TARGET, color='lightsalmon',
                                     fill=False, ls='--', linewidth=0.4)
        plt.gcf().gca().add_artist(draw_circle)
    plt.legend(fontsize=8)
    plt.grid(color='lightskyblue', linestyle='--', linewidth=0.3)
    ax.set_aspect(1)  # set the axes proportion
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    timestamp = dt.strftime(dt.now(), '%Y_%m_%d_%Hh%Mm%Ss')
    plt.savefig('figs/fig_' + timestamp + '.png', dpi=1000)
    plt.show()


χ = torch.zeros((NUM_OF_TARGETS * NUM_OF_UAVS, 1), dtype=torch.float32)  # Lagrange multiplier
b = torch.zeros((NUM_OF_TARGETS * NUM_OF_UAVS, 3), dtype=torch.float32)
µ = torch.zeros((NUM_OF_UAVS * (NUM_OF_UAVS - 1) // 2, 1), dtype=torch.float32)  # Lagrange multiplier
c = torch.zeros((NUM_OF_UAVS * (NUM_OF_UAVS - 1) // 2, 3), dtype=torch.float32)

omiga_b, omiga_c = 200, 200
# epsilon_1b, epsilon_1c, epsilon_2b, epsilon_2c = .95, .95, 1.05, 1.05
A2 = torch.FloatTensor(NUM_OF_UAVS * (NUM_OF_UAVS - 1) // 2, NUM_OF_UAVS)

rows = 0
for i in range(0, NUM_OF_UAVS - 1):
    for j in range(i + 1, NUM_OF_UAVS):
        A2[rows][j] = -1
        A2[rows][i] = 1
        rows = rows + 1


def delta_rb(q):
    return torch.norm(q.unfold(1, 3, 1).repeat(1, NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS,
                                                                                               1) - b)  # definition of norm is questionable


def delta_rc(q):
    return torch.norm(torch.matmul(A2, q) - c)


def absolute_of_biggest_element(X):
    length = X.shape[0]
    max_abs = 0
    for i in range(0, length):
        t = torch.norm(X[i])
        if t > max_abs:
            max_abs = t
    return max_abs


def P_admm(q_xy_temp, azimuth_temp):
    P_q_psai = sigma_gamma(q_xy_temp, azimuth_temp)
    q_temp = q_(q_xy_temp)
    return P_q_psai + ρ2 / 2 * torch.norm(torch.matmul(A2, q_temp) - c + µ).pow(2) - ρ2 / 2 * torch.norm(µ).pow(2) + \
        ρ1 / 2 * torch.norm(
            q_temp.unfold(1, 3, 1).repeat(1, NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1) - b + χ).pow(
            2) - \
        ρ1 / 2 * torch.norm(χ).pow(2)


def penalty1(q_xy_temp):
    q_temp = q_(q_xy_temp)
    return ρ2 / 2 * torch.norm(torch.matmul(A2, q_temp) - c + µ).pow(2)


def penalty2(q_xy_temp):
    q_temp = q_(q_xy_temp)
    return ρ1 / 2 * torch.norm(
        q_temp.unfold(1, 3, 1).repeat(1, NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1) - b + χ).pow(2)


MAX_ITERATION_TIMES = 200
MAX_ITERATION_TIMES_Q = 10
MAX_ITERATION_TIMES_AZI = 4
NUM_OF_SEARCH_DIRECTIONS = 10
original_alpha_q, original_alpha_azi = 12, .00001
threshold_q, threshold_azi = 0.00001, 0.00000001
ρ1, ρ2 = .01, .01  # penalty factor
ita = torch.tensor(0.000001)  # iteration threshold

# begin iteration
iter_times = 0
while 1:
    if iter_times < MAX_ITERATION_TIMES * 1 // 2:
        alpha_q, alpha_azi = original_alpha_q, original_alpha_azi
    elif iter_times < MAX_ITERATION_TIMES * 4 // 5:
        alpha_q, alpha_azi = original_alpha_q / 4, original_alpha_azi / 4
    else:
        alpha_q, alpha_azi = original_alpha_q / 16, original_alpha_azi / 16
    q = q_(q_xy)
    # update b
    for i in range(0, NUM_OF_UAVS):
        for j in range(0, NUM_OF_TARGETS):
            v = i * NUM_OF_TARGETS + j
            expression1 = q[i] - qt[j] + χ[v]
            ξ1 = torch.norm(expression1)
            b[v] = expression1 * torch.max(ξ1, MIN_DISTANCE_BETWEEN_UAV_AND_TARGET) / (ξ1 + 1e-7)
    # update c
    v = 0
    for i in range(0, NUM_OF_UAVS - 1):
        for j in range(i + 1, NUM_OF_UAVS):
            expression2 = q[i] - q[j] + µ[v]
            ξ2 = torch.norm(expression2)
            c[v] = expression2 * torch.max(ξ2, MIN_DISTANCE_BETWEEN_UAVS) / (ξ2 + 1e-7)
            v = v + 1

    # update q
    iter_times_q = 0
    q_grad_last, d_last = 0, 0
    beta_NAG = 0.
    while True:
        q_xy_temp = q_xy.clone()
        q_xy_temp.requires_grad = True

        # p1=penalty1(q_xy_temp)
        # p1.backward()
        # p1grad=q_xy_temp.grad.clone()
        # q_xy_temp.grad.zero_()
        #
        # p2=penalty2(q_xy_temp)
        # p2.backward()
        # p2grad = q_xy_temp.grad.clone()
        # q_xy_temp.grad.zero_()
        target_func = P_admm(q_xy_temp, azimuth)
        target_func.backward()
        targrad = q_xy_temp.grad.clone()
        d = beta_NAG * d_last + q_xy_temp.grad + beta_NAG * (q_xy_temp.grad - q_grad_last)
        if torch.norm(d) > 100:
            d = d * 100 / (torch.norm(d) + 1e-7)

        q_xy = q_xy - alpha_q * d
        for i in range(0, NUM_OF_UAVS):
            if q_xy[i][0] > SAFE_REGION_OF_UAVS:
                q_xy[i][0] = SAFE_REGION_OF_UAVS
        iter_times_q = iter_times_q + 1
        ju = torch.norm(q_xy_temp.grad)
        if torch.norm(q_xy_temp.grad) < threshold_q:
            break
        if iter_times_q >= MAX_ITERATION_TIMES_Q:
            break
        d_last = d
        q_grad_last = q_xy_temp.grad
        q_xy_temp.grad.zero_()
        del q_xy_temp
    q = q_(q_xy)
    # update azimuth
    iter_times_azimuth = 0
    # add random search,dispense dependency on initial value
    min_hori_angle, max_hori_angle = angle_range()
    for i in range(0, NUM_OF_UAVS):
        target_func_pre = sigma_gamma(q_xy, azimuth)
        for j in range(0, NUM_OF_SEARCH_DIRECTIONS):
            azimuth_pre = azimuth[i].clone()
            azimuth[i] = ((NUM_OF_SEARCH_DIRECTIONS - j - 1) * min_hori_angle[0][i] + j * max_hori_angle[0][i]) \
                         / (NUM_OF_SEARCH_DIRECTIONS - 1)
            # between min_hori_angle and max_hori_angle=angle_range(),get average dispersed directions
            target_func = sigma_gamma(q_xy, azimuth)
            if target_func < target_func_pre:
                target_func_pre = target_func
            else:
                azimuth[i] = azimuth_pre

    # grad descent
    while True:
        azimuth_temp = azimuth.clone()
        azimuth_temp.requires_grad = True
        target_func = sigma_gamma(q_xy, azimuth_temp)
        target_func.backward()
        azimuth = azimuth - alpha_azi * azimuth_temp.grad
        iter_times_azimuth = iter_times_azimuth + 1
        ju = torch.norm(azimuth_temp.grad)
        if torch.norm(azimuth_temp.grad) < threshold_azi:
            break
        if iter_times_azimuth > MAX_ITERATION_TIMES_AZI:
            break
        del azimuth_temp
    # project into -pi to pi
    for i in range(0, NUM_OF_UAVS):
        while True:
            if azimuth[i] > np.pi:
                azimuth[i] = azimuth[i] - 2 * np.pi
            elif azimuth[i] < -np.pi:
                azimuth[i] = azimuth[i] + 2 * np.pi
            else:
                break

    # update µ,χ
    µ = (µ + torch.matmul(A2, q) - c).detach()
    χ = (χ + q.unfold(1, 3, 1).repeat(1, NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS,
                                                                                      1) - b).detach()
    absolute_of_biggest_element_of_χ = absolute_of_biggest_element(χ)
    absolute_of_biggest_element_of_µ = absolute_of_biggest_element(µ)
    if absolute_of_biggest_element_of_χ > omiga_b:
        χ = χ / absolute_of_biggest_element_of_χ
    if absolute_of_biggest_element_of_µ > omiga_c:
        µ = µ / absolute_of_biggest_element_of_µ
    #  whether or not have converged

    iter_times = iter_times + 1
    if delta_rc(q) + delta_rb(q) < ita:
        print(time.time() - start_time)
        plot_fig()
        break
    if iter_times > MAX_ITERATION_TIMES:
        print(time.time() - start_time)
        plot_fig()
        break

    print(target_func)
    print(q_xy)
    print(azimuth)
