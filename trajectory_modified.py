import numpy as np  
import itertools

# Tp = 14e-6 
# Tc = 72e-6 
# T = 3*(Tp+Tc)
T=1

'''
include code to do object tracking in this class, take two frames, frame i and frame i-1 and detect common objects
for common objects, we need xj, yj from (i-1)th frame and rj from ith frame --> estimate function 

the estimate function will be running in a loop in main(), to find and store the new trajectory at each point 
'''

class Trajectory:

    def __init__(self):
        self.traj = np.empty((0,2), np.float32) #0 rows, 2 columns empty array

        #Add (0,0) to the array
        self.traj = np.vstack((self.traj, np.array([[0,0]])))

    def estimate(self, static_objects, vb):
        
        '''
        Code for extracting static objects in a frame and store them in an array in this form
        ---      ---
        |x1, y1, r1|
        |x2, y2, r2| 
        |    .     |
        |    .     |
        |xn, yn, rn|
        ---      ---
        '''
        Pr = self.localize(static_objects, vb)
        if Pr is None:
            pass
        else:
            self.traj = np.vstack((self.traj, Pr))

    def localize(self,static_objects, vb):
        #Number of points = NC2 = (N(N-1)/2)
        N = len(static_objects)
        intersections = np.empty((0,2), np.float32)

        for i, j in itertools.combinations(range(N), 2):
            circle1 = static_objects[i]
            circle2 = static_objects[j]
            intersection = self.localize_two_circles(circle1, circle2, vb)
            if intersection is None:
                pass
            else:
                intersections = np.vstack((intersections, intersection))

        #After finding all intersections, compute the mean
        return np.mean(intersections, axis=0)
    
    def circle_intersection(self, circle1, circle2):
        '''
        @summary: calculates intersection points of two circles
        @param circle1: tuple(x,y,radius)
        @param circle2: tuple(x,y,radius)
        @result: tuple of intersection points (which are (x,y) tuple)
        '''
        x1,y1,r1 = circle1
        x2,y2,r2 = circle2
        dx,dy = x2-x1,y2-y1

        P1 = np.array((x1, y1))
        P2 = np.array((x2, y2))
        Px1 = np.zeros(2)
        Px2 = np.zeros(2)
        d = np.linalg.norm(P1-P2)

        if d == 0:
            if r1==r2:
                # print("Coinciding circles, same radii, infinite sols")
                return None, None
            else:
                # print("Coinciding, no sols.")
                return None, None
        elif d < abs(r1-r2):
            # print("One circle inside other - no intersection, not possible")
            return None, None
        elif d == abs(r1-r2):
            #Internally touching 
            Px1 = (r1*P2 - r2*P1)/(r1-r2)
            Px2 = Px1 
        
        elif d > abs(r1-r2) and d < (r1+r2):
            print("2 points of intersection")
            a = (r1*r1-r2*r2+d*d)/(2*d)
            h = np.sqrt(r1*r1-a*a)
            xm = x1 + a*dx/d
            ym = y1 + a*dy/d
            xs1 = xm + h*dy/d
            xs2 = xm - h*dy/d
            ys1 = ym - h*dx/d
            ys2 = ym + h*dx/d 
            Px1 = np.array((xs1, ys1))
            Px2 = np.array((xs2, ys2))

        elif d == (r1+r2):
            Px1 = (r2*P1 + r1*P2)/(r1+r2)
            Px2 = Px1

        else:
            #Not touching, still we can estimate
            Px1 = (r2*P1 + r1*P2)/(r1+r2)
            Px2 = Px1

        print(f"Points of intersection : {Px1, Px2}")
        return Px1, Px2

    def localize_two_circles(self,circle1, circle2, vb):
        '''
        @summary: finds out the next coordinate of the bot
        @param circle1: tuple(x,y,radius)
        @param circle2: tuple(x,y,radius)
        @param vb: bot velocity 
        @result: tuple(xr, yr) of bot position
        ''' 

        Px1, Px2 = self.circle_intersection(circle1, circle2)
        if Px1 is None and Px2 is None:
            return None

        if np.array_equal(Px1, Px2):
            return Px1
        else:
            # I think this logic is wrong, we should be storing the past points
            # d1 = np.linalg.norm(Px1)
            # d2 = np.linalg.norm(Px2)
            d1 = np.linalg.norm(Px1 - self.traj[-1])
            d2 = np.linalg.norm(Px2 - self.traj[-1])

            print('Printing d1 and d2')
            print(d1,d2)
            e1 = abs(d1-vb*T)
            e2 = abs(d2-vb*T)
            if e1<e2:
                return Px1
            else:
                return Px2

    


if __name__ == "__main__":
    pass
