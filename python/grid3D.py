import numpy as np
import math

def buildTransformationBlock(R_ij):
    """
    Build a 6x6 block matrix for 'transforming' a 6x6 covariance
    from local edge coordinates to global coordinates.

    For a simple approach:
      block = [[R_ij,      0     ],
               [ 0,   R_ij_angles]]
    but we might just reuse R_ij for the rotational block.  A 
    more rigorous approach to Euler angles would consider partial
    derivatives w.r.t. roll/pitch/yaw. For a simplified approach,
    we treat orientation increments like a vector in R^3 as well.

    So we'll do:
      block = [[ R_ij, 0 ],
               [ 0,   R_ij ]]
    which is *not* strictly correct for Euler angles, but is often used
    in approximate pose-graph frameworks.
    """
    block = np.zeros((6,6))
    # top-left => R_ij for translation
    block[0:3, 0:3] = R_ij
    # bottom-right => R_ij for orientation
    block[3:6, 3:6] = R_ij
    return block

def posegraphMeasurementMatrix_3D(graph):
    """
    
    Expects graph.format == '3d'.
    Expects graph.edges to have shape (n_edges, 44) if storing:
      [id_i, id_j, dx, dy, dz, droll, dpitch, dyaw, flattened_6x6_info(36 floats)]
    
    Returns a list of dicts, one for each edge, with fields:
      - i, j  (node IDs)
      - t     (3D translation [dx, dy, dz])
      - R     (3x3 rotation from [droll, dpitch, dyaw])
      - Omega (6x6 'transformed' info matrix, if desired)
      - tau, kappa (some measure of translational / rotational weighting)
      - weight = 1 (placeholder)
    """
    if graph.get('format','3d').lower() != '3d':
        raise ValueError("graph.format must be '3d' for posegraphMeasurementMatrix_3D.")

    edges = graph['edges']
    n_edges = edges.shape[0]

    measurements = []
    for e_idx in range(n_edges):
        edge = edges[e_idx,:]
        i_idx = edge[0]
        j_idx = edge[1]

        # The 6D relative pose in columns [2..7]:
        dx, dy, dz, droll, dpitch, dyaw = edge[2:8]

        # Build the rotation from Euler angles:
        R_ij = eul2rotZYX(droll, dpitch, dyaw)
        t_ij = np.array([dx, dy, dz])

        # The next 36 entries define the 6x6 info matrix:
        # columns [8..44)
        Omega_6x6 = edge[8:44].reshape(6,6)

        # If you want to transform Omega to "global" coords 
        # (analogous to 2D code block^T * Omega * block),
        # create a 6x6 block matrix from R_ij. 
        block = buildTransformationBlock(R_ij) 
        Omega_transformed = block.T @ Omega_6x6 @ block

        # Check eigenvalues:
        eigvals = np.linalg.eigvals(Omega_transformed)
        if np.min(eigvals) < 0:
            print(f"Warning: negative eigenvalue in full covariance for edge {e_idx}, min eig:", np.min(eigvals))

        # Define a 3D analog of tau, kappa if you wish:
        # e.g. tau = 3.0 / trace(inv(OmegaTrans[0:3,0:3]))
        # kappa = something for the rotational part
        # This is  somewhat ad-hoc. We mimic the 2D logic:
        transBlock = Omega_transformed[0:3, 0:3]
        invTransBlock = np.linalg.inv(transBlock)
        tau_3d = 3.0 / np.trace(invTransBlock)

        rotBlock = Omega_transformed[3:6, 3:6]
        kappa_3d = rotBlock[2,2]  # as a placeholder

        meas = {
            'i': i_idx,
            'j': j_idx,
            # The main measurement pieces:
            't': t_ij,        # translation
            'R': R_ij,        # rotation
            'Omega': Omega_transformed,
            # We define 'tau' and 'kappa' as some scalar measure 
            # of the translational / rotational "confidence"
            'tau': tau_3d,
            'kappa': kappa_3d,
            'weight': 1
        }
        measurements.append(meas)

    return measurements

def poseAdd_3D(p_j, delta_pose):
    """
    3D version of poseAdd:
      p_out = p_j ⊕ delta_pose
    where p_j, delta_pose are each [x, y, z, roll, pitch, yaw].
    """
    # Convert each to np.array of shape (6,)
    p_j = np.array(p_j, dtype=float).reshape(6,)
    delta_pose = np.array(delta_pose, dtype=float).reshape(6,)

    # Unpack them
    x_j, y_j, z_j, roll_j, pitch_j, yaw_j = p_j
    dx, dy, dz, droll, dpitch, dyaw = delta_pose

    # Build rotation for p_j
    R_j = eul2rotZYX(roll_j, pitch_j, yaw_j)

    # The new orientation angles:
    new_roll  = roll_j  + droll
    new_pitch = pitch_j + dpitch
    new_yaw   = yaw_j   + dyaw
    # You can wrap angles to (-pi, pi] if you like:
    # new_roll = wrapAngle(new_roll) # etc.

    # The new position: p_j.xyz + R_j * [dx, dy, dz]
    new_xyz = np.array([x_j, y_j, z_j]) + R_j @ np.array([dx, dy, dz])

    return np.array([new_xyz[0], new_xyz[1], new_xyz[2],
                     new_roll, new_pitch, new_yaw])

def odometryFromEdges_3D(edges, nrNodes, verbosity=0, initialGuess=None):
    """
    3D version of odometryFromEdges, processing edges that connect consecutive
    nodes (1->2, 2->3, ..., nrNodes-1->nrNodes). The first (nrNodes-1) edges
    must be a spanning path in order, storing [dx, dy, dz, droll, dpitch, dyaw].
    
    edges : np.ndarray of shape (m, >=8)
       columns [0,1]   => node IDs (1-based)
       columns [2..7] => 6D relative poses
    nrNodes : int
    verbosity : int
    initialGuess : optional initial guess array of shape (nrNodes, 6)
    
    Returns
    -------
    poses : np.ndarray of shape (nrNodes, 6)
        The chain of poses built from odometry edges.
    """
    # We'll store a 6D pose for each node: [x,y,z,roll,pitch,yaw]
    poses = np.zeros((nrNodes, 6))

    # Construct the chain from node 1->2->3->...->nrNodes
    # in 1-based indexing. For each k in [2..nrNodes], the row is k-2 in 0-based indexing:
    for k in range(2, nrNodes + 1):
        row = k - 2
        # Check that edges[row] indeed connects node (k-1)->k in 1-based indexing
        # i.e. edges[row,0]==(k-1), edges[row,1]==k
        if not (edges[row, 1] == edges[row, 0] + 1 and edges[row, 0] == (k - 1)):
            print("Edge mismatch:", edges[row, :], "at row", row)
            raise ValueError("wrong odometric edges for a consecutive chain")

        # The 6D relative pose in columns [2..7]
        delta_pose = edges[row, 2:8]  # [dx, dy, dz, droll, dpitch, dyaw]

        # Compose poses[k-1] = poseAdd_3D(poses[k-2], delta_pose)
        #   (k-1, k-2 are 0-based indices in 'poses')
        poses[k - 1, :] = poseAdd_3D(poses[k - 2, :], delta_pose)

    if verbosity > 0:
        print("Initial guess from odometry (not plotting).")

    if initialGuess is not None:
        # Check if the user-supplied guess anchors the first node at the origin
        if np.linalg.norm(initialGuess[0,:]) > 1e-14:
            print("Provided initial guess does not have the first node at the origin")

    return poses

def anchorFirstNode_3D(posesGT):
    """
    Anchor the first node to the identity by subtracting the first pose
    from every subsequent pose, using a 3D "poseSubNoisy_3D" function.
    
    posesGT : numpy.ndarray of shape (num_nodes, 6)
              Each row is [x, y, z, roll, pitch, yaw].
    """
    anchor = posesGT[0, :].copy()  # The first node's 3D pose
    for i in range(posesGT.shape[0]):
        # Subtract anchor from posesGT[i], but with zero noise (sigma=0).
        posesGT[i, :] = poseSubNoisy_3D(posesGT[i, :], anchor, 0.0, 0.0, False)
    return posesGT

def findUndirectedEdge_3D(edge, edges_array):
    """
    Checks if 'edge' (a two-element list or array [id1, id2]) 
    already exists in edges_array in an undirected sense 
    ([id1, id2] or [id2, id1]). Returns a 1-based index if 
    found, else 0.
    """
    id1, id2 = edge
    for idx, e in enumerate(edges_array):
        if (e[0] == id1 and e[1] == id2) or (e[0] == id2 and e[1] == id1):
            return idx + 1  # mimic MATLAB's 1-based
    return 0

def wrapAngle(angle):
    """
    Wraps a single angle (in radians) to the interval (-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def eul2rotZYX(roll, pitch, yaw):
    """
    Build a 3x3 rotation matrix from Z-Y-X Euler angles.

    Convention (yaw -> pitch -> roll):
      Rz(yaw) * Ry(pitch) * Rx(roll)

    roll  = rotation about x
    pitch = rotation about y
    yaw   = rotation about z
    """
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,   0,  1]], dtype=float)
    Ry = np.array([[cp,  0, sp],
                   [0,   1, 0 ],
                   [-sp, 0, cp]], dtype=float)
    Rx = np.array([[1,   0,   0 ],
                   [0,   cr, -sr],
                   [0,   sr,  cr]], dtype=float)

    return Rz @ Ry @ Rx

def rotZYX2eul(R):
    """
    Extract Z-Y-X Euler angles (roll, pitch, yaw) from a 3x3 rotation matrix R.
    Returns (roll, pitch, yaw).
    """
    # clamp numeric issues
    R[2,0] = np.clip(R[2,0], -1.0, 1.0)

    pitch = -math.asin(R[2,0])
    roll  = math.atan2(R[2,1], R[2,2])
    yaw   = math.atan2(R[1,0], R[0,0])

    return (roll, pitch, yaw)

def poseSubNoisy_3D(p_j, p_i, sigmaT=0.0, sigmaR=0.0, isUniform=False):
    """
    3D version of poseSubNoisy, with poses p_i = [x_i, y_i, z_i, roll_i, pitch_i, yaw_i].
    deltaPose = inv(p_i) * p_j + noise => [dx, dy, dz, droll, dpitch, dyaw].
    """
    # Reshape to (6,) for safety
    p_j = np.array(p_j, dtype=float).reshape(6,)
    p_i = np.array(p_i, dtype=float).reshape(6,)

    # Extract translation + rotation
    xyz_i = p_i[:3]
    roll_i, pitch_i, yaw_i = p_i[3:6]

    xyz_j = p_j[:3]
    roll_j, pitch_j, yaw_j = p_j[3:6]

    # Build rotation matrices
    R_i = eul2rotZYX(roll_i, pitch_i, yaw_i)
    R_j = eul2rotZYX(roll_j, pitch_j, yaw_j)

    # Relative translation
    delta_xyz = R_i.T @ (xyz_j - xyz_i)
    # Relative rotation
    R_diff = R_i.T @ R_j
    droll, dpitch, dyaw = rotZYX2eul(R_diff)

    # Base relative pose
    deltaPose = np.array([delta_xyz[0], delta_xyz[1], delta_xyz[2],
                          droll, dpitch, dyaw])

    # Add noise
    if not isUniform:
        # Gaussian
        noise_xyz = sigmaT * np.random.randn(3)
        noise_rpy = sigmaR * np.random.randn(3)
    else:
        # Uniform
        noise_xyz = sigmaT * (2.0*np.random.rand(3) - 1.0)
        noise_rpy = sigmaR * (2.0*np.random.rand(3) - 1.0)

    deltaPoseNoisy = deltaPose + np.hstack((noise_xyz, noise_rpy))
    # Wrap angles
    deltaPoseNoisy[3] = wrapAngle(deltaPoseNoisy[3])
    deltaPoseNoisy[4] = wrapAngle(deltaPoseNoisy[4])
    deltaPoseNoisy[5] = wrapAngle(deltaPoseNoisy[5])

    return deltaPoseNoisy

def build_6x6_information_matrix(sigmaT, sigmaR):
    """
    Diagonal for independent translation + rotation.
    """
    info = np.zeros((6,6))
    # translation block
    info[0,0] = 1.0 / sigmaT**2
    info[1,1] = 1.0 / sigmaT**2
    info[2,2] = 1.0 / sigmaT**2
    # rotation block
    info[3,3] = 1.0 / sigmaR**2
    info[4,4] = 1.0 / sigmaR**2
    info[5,5] = 1.0 / sigmaR**2
    return info

def grid_random_graph_3D(nNodes,
                         Noise='gaussian',
                         RotationStd=0.01,
                         Scale=1.0,
                         TranslationStd=0.1,
                         LoopClosureProbability=0.3):
    """
    Builds a 3D "grid" random graph (assuming nNodes ~ perfect cube).
    Each node: [x,y,z,roll,pitch,yaw].
    Edges: row format => i, j, dx, dy, dz, droll, dpitch, dyaw, + flattened 6x6 info.
    """
    nrX = int(round(nNodes ** (1.0/3.0)))
    nrY = nrX
    nrZ = nrX
    if nrX*nrY*nrZ != nNodes:
        raise ValueError("For simplicity, this example requires nNodes to be a perfect cube.")

    sigmaT = TranslationStd
    sigmaR = RotationStd
    graph_scale = Scale
    probLC = LoopClosureProbability
    epsilon = 0.01

    # Build ground-truth
    posesGT = np.zeros((nNodes, 6))
    idx = 0
    for ix in range(nrX):
        for iy in range(nrY):
            for iz in range(nrZ):
                if (ix==0 and iy==0 and iz==0):
                    # anchor first node
                    roll = pitch = yaw = 0.0
                else:
                    roll  = (2.0*np.random.rand()-1.0)*np.pi
                    pitch = (2.0*np.random.rand()-1.0)*np.pi
                    yaw   = (2.0*np.random.rand()-1.0)*np.pi
                posesGT[idx,:] = [ix, iy, iz, roll, pitch, yaw]
                idx+=1

    # scale x,y,z
    posesGT[:,0:3] *= graph_scale

    # We'll do a chain of nNodes-1 edges for "odometry".
    # Then store them in a big array with 44 columns:
    # [id1,id2, dx,dy,dz, droll,dpitch,dyaw, + flattened 6x6 info].
    m = nNodes - 1
    edges = []

    for i in range(1, nNodes):
        row = np.zeros(44, dtype=float)
        id1 = i
        id2 = i+1
        row[0:2] = [id1, id2]
        p1 = posesGT[id1-1,:]
        p2 = posesGT[id2-1,:]
        delta_ij = poseSubNoisy_3D(p2, p1, sigmaT, sigmaR, (Noise.lower()=='uniform'))
        # fill the 6D relative pose
        row[2:8] = delta_ij
        # build a 6x6 info matrix and flatten
        info66 = build_6x6_information_matrix(sigmaT, sigmaR)
        row[8:44] = info66.flatten()
        edges.append(row)

    edges = np.array(edges)

    # Loop Closure
    edges_list = edges.tolist()  # convert array to Python list for appending
    m = edges.shape[0]           # current number of edges

    for id1 in range(2, nNodes+1):
        p1 = posesGT[id1-1,:]  # [x1, y1, z1, roll1, pitch1, yaw1]
        for id2 in range(id1+1, nNodes+1):
            p2 = posesGT[id2-1,:]
            
            # 3D distance
            dist_3d = np.linalg.norm(p1[0:3] - p2[0:3])
            
            if (dist_3d < (graph_scale + epsilon)) and (dist_3d > epsilon):
                # Check if edge already exists (undirected)
                e = [id1, id2]
                if np.random.rand() < probLC and (findUndirectedEdge_3D(e, np.array(edges_list)) == 0):
                    
                    # Build a new row for the loop closure
                    # We assume 44 columns: 
                    #   0..1 -> (id1, id2)
                    #   2..7 -> (dx, dy, dz, droll, dpitch, dyaw)
                    #   8..43-> flattened 6x6 info
                    row = np.zeros(44, dtype=float)
                    row[0:2] = [id1, id2]
                    
                    # 1) Compute the relative pose in 3D
                    delta_ij = poseSubNoisy_3D(
                        p2, p1, 
                        sigmaT=sigmaT, 
                        sigmaR=sigmaR, 
                        isUniform=(Noise.lower()=='uniform')
                    )
                    row[2:8] = delta_ij  # [dx,dy,dz, droll,dpitch,dyaw]
                    
                    # 2) Build or retrieve a 6x6 info matrix (diagonal example)
                    info66 = build_6x6_information_matrix(sigmaT, sigmaR)
                    row[8:44] = info66.flatten()

                    edges_list.append(row)
                    m += 1

    edges = np.array(edges_list)

    posesGT = anchorFirstNode_3D(posesGT)
    pose_estimate = odometryFromEdges_3D(edges, nNodes)

    graph = {
        'model': 'grid-3d',
        'format': '3d',
        'scale': graph_scale,
        'edges': edges,
        'poses_gt': posesGT,
        'pose_estimate': pose_estimate,  # if you want to do "odometryFromEdges_3D"
        'sigma_R': sigmaR,
        'sigma_t': sigmaT
    }

    meas = posegraphMeasurementMatrix_3D(graph)
    graph['measurements'] = meas

    return graph

def eulerAnglesToQuaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (in radians) to a quaternion [qx, qy, qz, qw].
    roll  = rotation around x
    pitch = rotation around y
    yaw   = rotation around z
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    
    return qx, qy, qz, qw


def exportGraphToG2o_3D(graph, filename, usePoseEstimate=True, fixFirstNode=True):
    """
    Exports the given 3D pose-graph to a .g2o file in SE3:QUAT format.
    
    :param graph: Dictionary as returned by `grid_random_graph_3D`.
                  It must have:
                  - 'poses_gt' or 'pose_estimate' in shape (nNodes,6).
                  - 'edges' in shape (m,44). Columns:
                        [0] -> i, [1] -> j,
                        [2..7] -> (dx, dy, dz, droll, dpitch, dyaw),
                        [8..43] -> flattened 6x6 info matrix in row-major order.
    :param filename: Name of the output text file to write.
    :param usePoseEstimate: If True, use graph['pose_estimate'] for the nodes.
                            Otherwise, use graph['poses_gt'].
    :param fixFirstNode: If True, add a line `FIX 1` after the first vertex 
                         so that node #1 is held fixed in the optimization.
    """
    # Choose which node poses to export (GT or estimate)
    if usePoseEstimate and ('pose_estimate' in graph):
        nodePoses = graph['pose_estimate']
    else:
        nodePoses = graph['poses_gt']
    
    nNodes = nodePoses.shape[0]
    
    edges = graph['edges']
    m = edges.shape[0]
    
    with open(filename, 'w') as f:
        #------------------------------------------------
        # Write VERTEX lines:
        #   VERTEX_SE3:QUAT <node_id>  x y z  qx qy qz qw
        #------------------------------------------------
        for node_idx in range(nNodes):
            # Our nodes are 1-based, so the node ID is (node_idx+1)
            node_id = node_idx + 1
            
            x, y, z, roll, pitch, yaw = nodePoses[node_idx, :]
            
            qx, qy, qz, qw = eulerAnglesToQuaternion(roll, pitch, yaw)
            
            f.write(f"VERTEX_SE3:QUAT {node_id} {x:.9f} {y:.9f} {z:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        
        # Optionally fix the first node
        #if fixFirstNode:
           # f.write(f"FIX 1\n")
        
        #------------------------------------------------
        # Write EDGE lines:
        #   EDGE_SE3:QUAT i j  dx dy dz  qx qy qz qw  <info_6x6_upper_tri>
        #------------------------------------------------
        for edge_idx in range(m):
            row = edges[edge_idx, :]
            i = int(row[0])
            j = int(row[1])
            
            dx, dy, dz, droll, dpitch, dyaw = row[2:8]
            
            # Convert relative Euler angles to quaternion:
            qx, qy, qz, qw = eulerAnglesToQuaternion(droll, dpitch, dyaw)
            
            # Flatten the 6x6 info matrix and pick only the UPPER triangle (21 entries in row-major)
            info_6x6 = row[8:44].reshape((6,6))
            info_upper_tri = []
            for r in range(6):
                for c in range(r, 6):
                    info_upper_tri.append(info_6x6[r,c])
            
            # Write the edge line
            f.write(f"EDGE_SE3:QUAT {i} {j} "
                    f"{dx:.9f} {dy:.9f} {dz:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f} ")
            f.write(" ".join(f"{val:.9f}" for val in info_upper_tri))
            f.write("\n")

    print(f"Graph exported to: {filename}")


def random_3D_test(s=10,problc=1.0, rotstd = 0.01, transtd = 0.01):
    """
    A 3D equivalent of random_2D_test:
      - Builds a random 3D grid graph with specified noise,
      - Creates the ground-truth rotation blocks in a big matrix R_gt,
      - Checks that the first rotation is identity,
      - Prints graph stats.
    
    Note:
      - 'grid_random_graph_3D' is assumed to be a function that returns:
          graph['poses_gt'] => shape (nrNodes, 6), 
                               each row [x, y, z, roll, pitch, yaw]
          graph['edges']    => the 3D edges array
      - 'eul2rotZYX' is a function that builds a 3x3 rotation from Euler angles.
    """
    # 3D dimension

    d = 3
    nrNodes = s*s*s      # e.g. a perfect cube: 4x4x4
    probLC = problc       # probability of loop closures
    rotStd = rotstd      # rotation noise std
    tranStd = transtd     # translation noise std

    # Build a random 3D grid graph (assuming grid_random_graph_3D is defined)
    graph = grid_random_graph_3D(
        nNodes=nrNodes,
        RotationStd=rotStd,
        TranslationStd=tranStd,
        Scale=1.0,
        LoopClosureProbability=probLC
    )
    nrEdges = graph['edges'].shape[0]

    # Build the ground-truth rotation blocks into a big matrix R_gt
    # shape: (3*nrNodes, 3). The i-th 3x3 block is the rotation from
    # [roll, pitch, yaw].
    R_gt = np.zeros((d * nrNodes, d))
    for i in range(nrNodes):
        roll  = graph['poses_gt'][i, 3]
        pitch = graph['poses_gt'][i, 4]
        yaw   = graph['poses_gt'][i, 5]
        R_3x3 = eul2rotZYX(roll, pitch, yaw) 
        # Insert into row slice for node i
        # row indices = 3*i : 3*i+3
        R_gt[3*i : 3*i+3, :] = R_3x3

    # Check the first 3x3 block for identity
    first_block = R_gt[0:3, 0:3]
    if np.linalg.norm(first_block - np.eye(3)) > 1e-6:
        raise ValueError("First rotation != identity")

    print(f"Random 3D grid graph: number of nodes: {nrNodes}, number of edges: {nrEdges}.")

    plot_3D_graph(graph)
    return graph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3D_graph(graph):
    """
    Plots a 3D grid random graph:
      - ground-truth node positions in 3D
      - edges between nodes
      - the estimated trajectory in 3D
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))  # make aspect ratio equal

    poses_gt  = graph['poses_gt']      # shape (N, 6): [x, y, z, roll, pitch, yaw]
    pose_est  = graph.get('pose_estimate', None)  # shape (N, 6) or None
    edges     = graph['edges']

    # Plot ground-truth node positions
    ax.scatter(poses_gt[:,0], poses_gt[:,1], poses_gt[:,2], 
               c='b', marker='o', s=20, label='Ground Truth Nodes')

    # Plot edges
    for i in range(edges.shape[0]):
        # edges[i,0], edges[i,1] => node IDs in 1-based indexing
        id1 = int(edges[i,0]) - 1
        id2 = int(edges[i,1]) - 1
        p1 = poses_gt[id1, 0:3]
        p2 = poses_gt[id2, 0:3]
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]
        ax.plot(xs, ys, zs, 'k-', linewidth=1)

    # Plot estimated trajectory if available
    if pose_est is not None:
        ax.plot(pose_est[:,0], pose_est[:,1], pose_est[:,2], 
                'r--', linewidth=2, label='Estimated Trajectory')

    ax.set_title('3D Grid Random Graph')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.show()

if __name__ == "__main__":
    s = 3
    exportGraphToG2o_3D(random_3D_test(s=s),f'grid{s}D.g2o')



