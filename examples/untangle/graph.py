from random import randint
from pyglet.gl import *

from pymt import *



def drawVertex(x,y):
       glPushMatrix()
       glTranslated(x,y, 0)
       glColor3d(1.0,1.0,1.0)
       gluDisk(gluNewQuadric(), 0, 25, 32,1)
       glScaled(0.75,0.75,1.0)
       glColor3d(0.2,0.6,0.2)
       gluDisk(gluNewQuadric(), 0, 25, 32,1)
       glPopMatrix()

def drawCollision(x,y):
       glPushMatrix()
       glTranslated(x,y-5, 0)
       with gx_blending:
              glColor4f(1.0,0.0,0.0, 0.3)
              drawTriangle(pos=(0,0),w=20,h=20)
       #gluDisk(gluNewQuadric(), 0, 10, 32,1)
       glPopMatrix()


def point_inside_line_segment(point, p1, p2):
       minx = min(p1.x, p2.x)
       miny = min(p1.y, p2.y)
       maxx = max(p1.x, p2.x)
       maxy = max(p1.y, p2.y)
       #print minx, maxx, miny, maxy, point.x, point.y
       if point.x > minx and point.x < maxx and point.y > miny and point.y < maxy:
              return True

class Graph(object):
       def __init__(self, num_verts=12, displaySize=(640,480)):
              self.verts = []
              for i in range(num_verts):
                     x = randint(100,displaySize[0]-100)*1.0
                     y = randint(100,displaySize[1]-100)*1.0
                     self.verts.append([x,y])
              
              self.edges = [ [self.verts[i], self.verts[(i+1)%num_verts]] for i in range(num_verts) ]
              self.collisions = []
              self.is_solved()
              
       def is_solved(self):
              self.collisions = []
              for e1 in self.edges:
                     for e2 in self.edges:
                            if  e1 != e2:
                                   p1,p2,p3,p4 = Vector(*e1[0]), Vector(*e1[1]), Vector(*e2[0]), Vector(*e2[1])
                                   intersection = Vector.line_intersection( p1,p2,p3,p4 )
                                   if (Vector.distance(intersection, p1) > 0.2 and
                                       Vector.distance(intersection, p2) > 0.2 and
                                       Vector.distance(intersection, p3) > 0.2 and
                                       Vector.distance(intersection, p4) > 0.2 and
                                       point_inside_line_segment(intersection, p1,p2) and
                                       point_inside_line_segment(intersection, p3,p4)):
                                          self.collisions.append(intersection)
              return len(self.collisions) == 0
     
       def draw(self):
              #self.is_solved()
              for e in self.edges:
                     glColor3d(1,1,1)
                     drawLine((e[0][0],e[0][1], e[1][0],e[1][1]), width=12.0)
                     glColor3d(0.3,0.6,0.3)
                     drawLine((e[0][0],e[0][1], e[1][0],e[1][1]), width=6.0)
              for v in self.verts:
                     drawVertex(v[0],v[1])
              for c in self.collisions:
                     drawCollision(c.x,c.y)
                     
       #returns the vertex at the position, None if no vertex there
       def collideVerts(self, x,y, regionSize=40):
              for v in self.verts:
                     dx = abs(x - v[0])
                     dy = abs(y - v[1])
                     if (dx < regionSize and dy < regionSize):
                         return v
              return None


if __name__ == "__main__":
	print "this is an implementation file only used by untabgle.py"
