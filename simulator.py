import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.collections as mc

def pause_plot():
    rad2deg = 180 / math.pi
    deg2rad = math.pi / 180
    kmh2ms = 1000 / 3600

    interval = 0.1
    time = 0
    end_time = 5

    x = 0  # X coordinate of the center of the rear wheel shaft
    y = 0  # Y coordinate of the center of the rear wheel shaft
    v = 10 * kmh2ms # velocity of the center of the rear wheel shaft
    yaw = 0.1
    steer = 0
    wheelbase = 2.8
    length = 4.985
    width = 1.845

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-5, 50])
    ax.set_ylim([-5, 50])
    ax.set(aspect=1)

    point, = ax.plot(x, y, "bo")

    body_center = [x + wheelbase / 2 * math.cos(yaw), y + wheelbase / 2 * math.sin(yaw)]

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
    
    lines, = ax.plot(body_x, body_y)

    while time < end_time:
        time = time + interval

        steer = math.sin(time) / 10

        yaw = yaw + v * math.tan(steer) * interval
        x = x + v * math.cos(yaw) * interval
        y = y + v * math.sin(yaw) * interval
        body_center = [x + wheelbase / 2 * math.cos(yaw), y + wheelbase / 2 * math.sin(yaw)]

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

        point.set_data([x,], [y,])
        lines.set_data(body_x, body_y)

        plt.pause(0.1)

if __name__ == "__main__":
    pause_plot()