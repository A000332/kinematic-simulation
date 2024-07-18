import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import csv
import cvxpy as cvx

def pause_plot():
    ## const
    velocity = 10
    NP = 3 # prection horizon
    print("NP:", NP)
    state_dim = 4 # dimension of xt
    control_dim = 2 # dimension of ut
    gain_Q_x = 1
    gain_Q_y = 150
    gain_Q_v = 10000
    gain_Q_yaw = 10000
    gain_R_accl = 1
    gain_R_yawrate = 1

    init_x = 0  # X coordinate of the center of the rear wheel shaft
    init_y = 0  # Y coordinate of the center of the rear wheel shaft
    init_v = velocity  # velocity of the center of the rear wheel shaft
    init_yaw = 0
    x0 = np.array([init_x, init_y, init_v, init_yaw]).T.reshape(4, -1)
    wheelbase = 2.8
    length = 4.985
    width = 1.845

    ## for reading data
    t_refs = []
    x_refs = []
    y_refs = []
    yaw_refs = []

    ## read sample trajectory
    with open('./ref_traj_sin.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 0:
                t_refs.append(float(row[0]))
                x_refs.append(float(row[1]))
                y_refs.append(float(row[2]))
                yaw_refs.append(float(row[3]))

    csv_data_len = len(t_refs)

    t_refs = np.array(t_refs)
    x_refs = np.array(x_refs)
    y_refs = np.array(y_refs)
    v_refs = [velocity] * csv_data_len
    v_refs = np.array(v_refs)
    yaw_refs = np.array(yaw_refs)

    Y_ref_list = []
    for x, y, v, yaw in zip(x_refs, y_refs, v_refs, yaw_refs):
        Y_ref_list.append(np.array([x, y, v, yaw]))

    ## read output reference
    accl_refs = []
    yawrate_refs = []

    # with open('./ref_output_jagged2.csv') as f_o:
    #     reader_o = csv.reader(f_o)
    #     for i, row in enumerate(reader_o):
    #         if i > 0:
    #             accl_refs.append(float(row[0]))
    #             yawrate_refs.append(float(row[1]))
    
    accl_refs = np.zeros(csv_data_len)
    yawrate_refs = np.zeros(csv_data_len)

    # initial plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-5, 60])
    ax.set_ylim([-10, 10])
    ax.set(aspect=1)
    ax.plot(x_refs, y_refs, color="orange") # plot referential trajectory

    point, = ax.plot(init_x, init_y, "bo")

    body_center = [init_x + wheelbase / 2 * math.cos(init_yaw), init_y + wheelbase / 2 * math.sin(init_yaw)]

    # rear left -> rear right -> front right -> front left -> rear left
    body_x = [body_center[0] - width / 2 * math.sin(init_yaw) - length / 2 * math.cos(init_yaw), \
                body_center[0] + width / 2 * math.sin(init_yaw) - length / 2 * math.cos(init_yaw), \
                body_center[0] + width / 2 * math.sin(init_yaw) + length / 2 * math.cos(init_yaw), \
                    body_center[0] - width / 2 * math.sin(init_yaw) + length / 2 * math.cos(init_yaw), \
                        body_center[0] - width / 2 * math.sin(init_yaw) - length / 2 * math.cos(init_yaw)]
    body_y = [body_center[1] + width / 2 * math.cos(init_yaw) - length / 2 * math.sin(init_yaw), \
                body_center[1] - width / 2 * math.cos(init_yaw) - length / 2 * math.sin(init_yaw), \
                body_center[1] - width / 2 * math.cos(init_yaw) + length / 2 * math.sin(init_yaw), \
                    body_center[1] + width / 2 * math.cos(init_yaw) + length / 2 * math.sin(init_yaw), \
                        body_center[1] + width / 2 * math.cos(init_yaw) - length / 2 * math.sin(init_yaw)]

    lines, = ax.plot(body_x, body_y)

    traj_x = np.array([init_x])
    traj_y = np.array([init_y])
    traj_pts, = ax.plot(traj_x, traj_y, "b")

    # open csv file for logging x_diff
    f = open('x1_diff.csv', 'w', newline='')
    writer = csv.writer(f)

    interval = 0.1
    time = 0
    end_time = interval * (csv_data_len - NP)

    itr = 0
    while time < end_time:
        print('\ntime = ' + f'{time:.1f}')

        Y_ref = Y_ref_list[itr]
        for i in range(itr + 1, itr + NP):
            Y_ref = np.hstack([Y_ref, Y_ref_list[i].T])
        Y_ref = Y_ref.reshape(len(Y_ref_list[0]) * NP, -1)

        ## Ak, Bk, Wk matrix
        Ak_list = []
        Bk_list = []
        Wk_list = []
        for i in range(itr, itr + NP):
            t_r = t_refs[i]
            x_r = x_refs[i]
            y_r = y_refs[i]
            yaw_r = yaw_refs[i]
            v_r = v_refs[i]
            dt = t_refs[1] - t_refs[0]
            
            Ak = np.array([[1, 0, math.cos(yaw_r) * dt, -v_r * math.sin(yaw_r) * dt],
                        [0, 1, math.sin(yaw_r) * dt, -v_r * math.cos(yaw_r) * dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
            
            Ak_list.append(Ak)

            Bk = np.array([[0, 0],
                        [0, 0],
                        [dt, 0],
                        [0, dt]])
            
            Bk_list.append(Bk)
            
            Wk = np.array([[-v_r * math.cos(yaw_r) + v_r * math.sin(yaw_r) * yaw_r],
                        [-v_r * math.sin(yaw_r) - v_r * math.cos(yaw_r) * yaw_r],
                        [0],
                        [0],
                        ])
            
            Wk_list.append(Wk)

        ## F, G, S, W matrix
        F = []
        G = []
        S = []

        Ak_pre = np.eye(Ak_list[0].shape[0])

        S_row_pre = []
        S_row_pre.append(np.eye(Ak_list[0].shape[0]))
        for j in range(1, NP):
            S_row_pre.append(np.zeros((Ak_list[0].shape[0], Ak_list[0].shape[1])))

        G_row_pre = []
        G_row_pre.append(Bk_list[0])
        for j in range(1, NP):
            G_row_pre.append(np.zeros((Bk_list[0].shape[0], Bk_list[0].shape[1])))

        for i in range(0, NP):
            F.append(Ak_list[i]@Ak_pre)
            Ak_pre = Ak_list[i]@Ak_pre
            for j in range(i):
                S_row_pre[j] = Ak_list[i]@S_row_pre[j]
                G_row_pre[j] = Ak_list[i]@G_row_pre[j]
            S_row_pre[i] = np.eye(Ak_list[0].shape[0])
            G_row_pre[i] = Bk_list[i]
            S.append(copy.deepcopy(S_row_pre))
            G.append(copy.deepcopy(G_row_pre))

        F_mat = F[0]
        S_mat = S[0][0]
        for j in range(1, NP):
            S_mat = np.hstack((S_mat, S[0][j]))
        G_mat = G[0][0]
        for j in range(1, NP):
            G_mat = np.hstack((G_mat, G[0][j]))

        W_mat = Wk_list[0]
        for i in range(1, NP):
            F_mat = np.vstack((F_mat, F[i]))
            S_mat_row = S[i][0]
            G_mat_row = G[i][0]
            W_mat = np.vstack((W_mat, Wk_list[i]))
            for j in range(1, NP):
                S_mat_row = np.hstack((S_mat_row, S[i][j]))
                G_mat_row = np.hstack((G_mat_row, G[i][j]))
            S_mat = np.vstack((S_mat, copy.deepcopy(S_mat_row)))
            G_mat = np.vstack((G_mat, copy.deepcopy(G_mat_row)))

        ## x0, H, C, Q, R, U_ref
        print("X0:", np.ravel(x0))
        H = np.eye(NP)
        C = np.eye(F_mat.shape[0])
        Q = np.zeros((state_dim * NP, state_dim * NP))
        for i in range(0, state_dim * NP):
            if i % state_dim == 0:
                Q[i][i] = gain_Q_x
            elif i % state_dim == 1:
                Q[i][i] = gain_Q_y
            elif i % state_dim == 2:
                Q[i][i] = gain_Q_v
            else:
                Q[i][i] = gain_Q_yaw

        output_ref = []
        for i in range(itr, itr + NP):
            output_ref.append(accl_refs[i])
            output_ref.append(yawrate_refs[i])
        U_ref = np.array(output_ref)
        R = np.zeros((control_dim * NP, control_dim * NP))
        for i in range(0, control_dim * NP):
            if i % control_dim == 0:
                R[i][i] = gain_R_accl
            else:
                R[i][i] = gain_R_yawrate

        ## M, N, U
        M = ((C @ G_mat).T) @ Q @ C @ G_mat + R
        N = (C@(F_mat@x0 + (S_mat @ W_mat))-Y_ref).T @ (Q @ C @ G_mat) - U_ref.T @ R

        u = cvx.Variable(control_dim * NP)
        obj = cvx.Minimize(u.T @ M @ u + 2 * N @ u)

        constraints = [u[1] <= 1, u[1] >= -1, u[3] <= 1, u[3] >= -1, u[5] <= 1, u[5] >= -1]
        prob = cvx.Problem(obj, constraints)
        #prob = cvx.Problem(obj)
        prob.solve()

        #U = -np.linalg.inv(M) @ N.T
        print("U:", u.value)
        #print("U_ref:", U_ref)
        print("U_diff:", u.value[0] - U_ref[0], u.value[1] - U_ref[1])

        ## update to new state
        x1 = np.array([x0[0] + (x0[2] + u.value[0] * dt) * math.cos(x0[3] + u.value[1] * dt) * dt, x0[1] + (x0[2] + u.value[0] * dt) * math.sin(x0[3] + u.value[1] * dt) * dt, x0[2] + u.value[0] * dt, x0[3] + u.value[1] * dt])
        print("X1:", np.ravel(x1))
        print("X1_ref:", Y_ref_list[itr + 1])
        print("X1_diff:", np.ravel(x1) - Y_ref_list[itr + 1])

        data = [time]
        data.extend(np.ravel(x1) - Y_ref_list[itr + 1])
        writer.writerow(data)

        body_center = [x1[0] + wheelbase / 2 * math.cos(x1[3]), x1[1] + wheelbase / 2 * math.sin(x1[3])]
        yaw = x1[3]

        # rear left -> rear right -> front right -> front left -> rear left
        body_x = [body_center[0] - width / 2 * math.sin(yaw) - length / 2 * math.cos(yaw), \
                body_center[0] + width / 2 * math.sin(yaw) - length / 2 * math.cos(yaw), \
                    body_center[0] + width / 2 * math.sin(yaw) + length / 2 * math.cos(yaw), \
                        body_center[0] - width / 2 * math.sin(yaw) + length / 2 * math.cos(yaw), \
                            body_center[0] - width / 2 * math.sin(yaw) - length / 2 * math.cos(yaw)]
        body_y = [body_center[1] + width / 2 * math.cos(yaw) - length / 2 * math.sin(yaw), \
                body_center[1] - width / 2 * math.cos(yaw) - length / 2 * math.sin(yaw), \
                    body_center[1] - width / 2 * math.cos(yaw) + length / 2 * math.sin(yaw), \
                        body_center[1] + width / 2 * math.cos(yaw) + length / 2 * math.sin(yaw), \
                            body_center[1] + width / 2 * math.cos(yaw) - length / 2 * math.sin(yaw)]

        ax.set_title('NP=' + str(NP) + ' Q_x=' + str(gain_Q_x)  + ' Q_y=' + str(gain_Q_y) + ' Q_v=' + str(gain_Q_v) + ' Q_yaw=' + str(gain_Q_yaw) + ' R_accl=' + str(gain_R_accl) + ' R_yr=' + str(gain_R_yawrate) + ' time = ' + f'{time:.1f}')
        point.set_data(x1[0], x1[1])
        lines.set_data(body_x, body_y)

        traj_x = np.append(traj_x, x1[0])
        traj_y = np.append(traj_y, x1[1])
        traj_pts.set_data(traj_x, traj_y)

        plt.pause(0.1)
        time = time + interval
        itr = itr + 1
        x0 = x1
    
    fig.savefig('NP=' + str(NP) + ' Q_x=' + str(gain_Q_x)  + ' Q_y=' + str(gain_Q_y) + ' Q_v=' + str(gain_Q_v) + ' Q_yaw=' + str(gain_Q_yaw) + ' R_accl=' + str(gain_R_accl) + ' R_yr=' + str(gain_R_yawrate) + '.png')
    f.close()

if __name__ == "__main__":
    pause_plot()