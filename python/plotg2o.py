import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

INPUT_FILE    = "/Users/nikolassanderson/Documents/GitHub/AlgoLib/newparking-garage.g2o"
ARROW_EVERY   = 10      # draw an arrow every N poses
ARROW_LENGTH  = 0.0     # arrow length

def quat_to_rot_matrix(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0: return np.eye(3)
    qx, qy, qz, qw = q / n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx), 1-2*(xx+yy)]
    ])

def load_vertices(filepath):
    idx2, xs2, ys2, th2 = [], [], [], []
    idx3, xs3, ys3, zs3, q3 = [], [], [], [], []

    with open(filepath, 'r') as f:
        for line in f:
            p = line.split()
            if len(p)==5 and p[0]=="VERTEX_SE2":
                _, i, x, y, th = p
                idx2.append(int(i)); xs2.append(float(x))
                ys2.append(float(y)); th2.append(float(th))

            elif len(p)==9 and p[0]=="VERTEX_SE3:QUAT":
                _, i, x, y, z, qx, qy, qz, qw = p
                idx3.append(int(i)); xs3.append(float(x))
                ys3.append(float(y)); zs3.append(float(z))
                q3.append([float(qx), float(qy), float(qz), float(qw)])

    # sort SE2 by index
    if idx2:
        order2 = np.argsort(idx2)
        xs2, ys2, th2 = np.array(xs2)[order2], np.array(ys2)[order2], np.array(th2)[order2]
    else:
        xs2 = ys2 = th2 = np.array([])

    # sort SE3 by index
    if idx3:
        order3 = np.argsort(idx3)
        xs3 = np.array(xs3)[order3]
        ys3 = np.array(ys3)[order3]
        zs3 = np.array(zs3)[order3]
        q3  = np.array(q3)[order3]
    else:
        xs3 = ys3 = zs3 = q3 = np.array([])

    return {'2d': (xs2, ys2, th2), '3d': (xs3, ys3, zs3, q3)}

def plot_trajectory_2d(xs, ys, thetas):
    plt.figure(figsize=(8,8))
    plt.plot(xs, ys, '-o', markersize=3, label="SE2 Path")
    for x, y, th in zip(xs[::ARROW_EVERY], ys[::ARROW_EVERY], thetas[::ARROW_EVERY]):
        dx, dy = ARROW_LENGTH*np.cos(th), ARROW_LENGTH*np.sin(th)
        plt.arrow(x, y, dx, dy,
                  head_width=0.2*ARROW_LENGTH,
                  head_length=0.2*ARROW_LENGTH,
                  length_includes_head=True,
                  color='C1')

    # enforce same axis range
    mn = min(xs.min(), ys.min())
    mx = max(xs.max(), ys.max())
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.gca().set_aspect('equal', 'box')

    plt.xlabel("X"); plt.ylabel("Y")
    plt.title("SE(2) Trajectory")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()

def set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xr, yr, zr = xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]
    r = max(xr, yr, zr)/2
    xm, ym, zm = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    ax.set_xlim3d(xm-r, xm+r)
    ax.set_ylim3d(ym-r, ym+r)
    ax.set_zlim3d(zm-r, zm+r)

def plot_trajectory_3d(xs, ys, zs, quats):
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, '-o', markersize=3, label="SE3 Path")
    for i in range(0, len(xs), ARROW_EVERY):
        R = quat_to_rot_matrix(quats[i])
        o = np.array([xs[i], ys[i], zs[i]])
        dir_vec = R @ np.array([1.0, 0.0, 0.0])
        ax.quiver(
            o[0], o[1], o[2],
            dir_vec[0], dir_vec[1], dir_vec[2],
            length=ARROW_LENGTH, normalize=True, color='C2'
        )

    # enforce equal scaling
    set_axes_equal(ax)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("SE(3) Trajectory")
    ax.legend(); plt.tight_layout(); plt.show()

def main():
    data    = load_vertices(INPUT_FILE)
    xs2, ys2, th2    = data['2d']
    xs3, ys3, zs3, q3 = data['3d']

    if xs3.size:
        plot_trajectory_3d(xs3, ys3, zs3, q3)
    elif xs2.size:
        plot_trajectory_2d(xs2, ys2, th2)
    else:
        print("No poses found in file.")

if __name__ == "__main__":
    main()