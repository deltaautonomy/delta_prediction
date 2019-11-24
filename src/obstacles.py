import numpy as np 
import math
import pdb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class Obstacle:

    def __init__(self, state, dim):
        self.origin = state[0:2]
        self.orientation = state[2]
        self.dimension = dim

        self.corners = np.zeros((4,2))
        self.axes = np.zeros((2,2))

        self.R = np.zeros((2,2))
        self.createCuboid()

    # Finds the corners and axes 
    def createCuboid(self):
        # Creates Rotation Matrix for cuboid 
        self.R = np.array([[math.cos(self.orientation), -math.sin(self.orientation)],
                           [math.sin(self.orientation),  math.cos(self.orientation)]])

        self.findCorners()

        self.findAxes()        


    def findCorners(self):
        # Corners of a cuboid of length one and orientation of zero along x,y and z
        direction = np.array([[ 0.5, 0.5],[-0.5, 0.5], \
                              [ 0.5,-0.5],[-0.5,-0.5]])

        # Dimension along x,y and z
        D = np.tile(self.dimension, (4, 1))

        # Cuboid scaled according to dimensions
        direction = direction*D

        # Origin of the cuboid
        O = np.tile(self.origin, (4,1))

        # Corners after rotation by R and translation by O
        self.corners = np.matmul(self.R, (direction).T).T + O

    def findAxes(self):
        # Axis of the cuboid before rotation
        direction = np.array([[1,0],[0,1]])

        # Rotation and normalization
        self.axes = np.matmul(self.R, direction).T
        self.axes = normalize(self.axes, axis=1)
    
    def getAxes(self):
        return self.axes

    def getCorners(self):
        return self.corners

    # Projection of the cuboid corners on an axis
    def project(self, axis):
        return np.matmul(self.corners, axis)

    def draw(self, ax, color='r'):
        # plt.close()
        self.corners[[2,3]] = self.corners[[3,2]]
        rect = matplotlib.patches.Polygon(self.corners[:,:], color=color, alpha=0.50)

        # angle = np.pi*self.orientation/180.0
        # t2 = matplotlib.transforms.Affine2D().rotate_deg(angle) + ax.transData
        # rect.set_transform(t2)

        ax.add_patch(rect)
        

# Finds of the 15 collision axes from the normals of the two cuboids
def createCollisionAxis(ax1, ax2):  
    axes = ax1
    axes = np.vstack((axes, ax2))

    return axes

# Finds the projecttion of the corners of the 2 cuboids on the Collision Axis
def checkProjection(box1, box2, axes):
    for axis in axes:
        proj1 = box1.project(axis)      
        proj2 = box2.project(axis)
        p1_min = proj1.min()
        p1_max = proj1.max()
        p2_min = proj2.min()
        p2_max = proj2.max()
    
        # Finds if there is a collision according to a particular axis
        if (p1_max < p2_min and p1_min < p2_min) or \
            (p2_max < p1_min and p2_min < p1_min):
                # print(axis)
                # print(box1.corners)
                # print(box2.corners)
                return False

    return True

# Main function with calls all the other collsion detection functions 
def collisionChecking(obj1, obj2):
    box1 = Obstacle(obj1[0], obj1[1])
    box2 = Obstacle(obj2[0], obj2[1])
    ax1 = box1.getAxes()
    ax2 = box2.getAxes()
    axes = createCollisionAxis(ax1, ax2)
    collision = checkProjection(box1, box2, axes)

    # fig,ax = plt.subplots(1)
    # box1.draw(ax, 'b')
    # box2.draw(ax)

    # ax.set_xlim(-3,3)
    # ax.set_ylim(-3,3)
    # plt.show()
    # plt.pause(0.001)

    return collision

if __name__ == "__main__":
    print("Doing Collision Checking")

    # x, y, orienation    dim_x dim_y
    ref_cube = [np.array([0.0, 0.0, 0.0]), np.array([1, 1])]
    test_set = [np.array([0.0, 0.75, np.pi/3]), np.array([1.0, 1.5])]

    print(collisionChecking(ref_cube, test_set))
    # r = Obstacle(ref_cube[0], ref_cube[1])
    # print(r.getAxes())


    # ref_cube = np.array([[0, 0, 0], [0, 0, 0], [3, 1, 2]])

    # test_set = np.array([[[ 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0], [ 0.8, 0.8, 0.8]], \
    #                      [[ 1.5,-1.5, 0.0], [ 1.0, 0.0, 1.5], [ 1.0, 3.0, 3.0]], \
    #                      [[ 0.0, 0.0,-1.0], [ 0.0, 0.0, 0.0], [ 2.0, 3.0, 1.0]], \
    #                      [[ 3.0, 0.0, 0.0], [ 0.0, 0.0, 0.0], [ 3.0, 1.0, 1.0]], \
    #                      [[-1.0, 0.0,-2.0], [ 0.5, 0.0, 0.4], [ 2.0, 0.7, 2.0]], \
    #                      [[ 1.8, 0.5, 1.5], [-0.2, 0.5, 0.0], [ 1.0, 3.0, 1.0]], \
    #                      [[ 0.0,-1.2, 0.4], [0.0,0.785,0.785],[ 1.0, 1.0, 1.0]], \
    #                      [[-0.8, 0.0,-0.5], [ 0.0, 0.0, 0.2], [ 1.0, 0.5, 0.5]]])
             
    # for i in range(test_set.shape[0]):
    #         collision = collisionChecking(ref_cube, test_set[i,:,:])
    #         print(i+1,collision)