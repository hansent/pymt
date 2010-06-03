from pymt import *
from _transformations import *


import numpy
from math import atan,cos
from OpenGL.GL import glMultMatrixf



def concatenate_matrices(*matrices):
    M = numpy.identity(4)
    for i in matrices:
        M = numpy.dot(M, i)
    return M




class MTScatter(MTWidget):
    '''MTScatter is a scatter widget based on MTWidget.
    You can scale, rotate and move with one and two finger.

    :Parameters:
        `rotation` : float, default to 0.0
            Set initial rotation of widget
        `translation` : list, default to (0,0)
            Set the initial translation of widget
        `scale` : float, default to 1.0
            Set the initial scaling of widget
        `do_rotation` : boolean, default to True
            Set to False for disabling rotation
        `do_translation` : boolean or list, default to True
            Set to False for disabling translation, and ['x'], ['y'] for limit translation only on x or y
        `do_scale` : boolean, default to True
            Set to False for disabling scale
        `auto_bring_to_front` : boolean, default to True
            Set to False for disabling widget bring to front
        `scale_min` : float, default to 0.01
            Minimum scale allowed. Don't set to 0, or you can have error with singular matrix.
            The 0.01 mean you can de-zoom up to 10000% (1/0.01*100).
        `scale_max` : float, default to None
            Maximum scale allowed.

    :Events:
        `on_transform` (rotation, scale, trans, intersect)
            Fired whenever the Scatter Widget is transformed (rotate, scale, moved, or zoomed).
    '''

    def __init__(self, **kwargs):

        kwargs.setdefault('rotation', 0.0)
        kwargs.setdefault('translation', (0,0))
        kwargs.setdefault('scale', 1.0)
        kwargs.setdefault('do_scale', True)
        kwargs.setdefault('do_rotation', True)
        kwargs.setdefault('do_translation', True)
        kwargs.setdefault('auto_bring_to_front', True)
        kwargs.setdefault('scale_min', 0.01)
        kwargs.setdefault('scale_max', None)

        super(MTScatter, self).__init__(**kwargs)

        self.register_event_type('on_transform')

        self.auto_bring_to_front = kwargs.get('auto_bring_to_front')
        self.scale_min      = kwargs.get('scale_min')
        self.scale_max      = kwargs.get('scale_max')
        self.do_scale       = kwargs.get('do_scale')
        self.do_rotation    = kwargs.get('do_rotation')
        self.do_translation = kwargs.get('do_translation')
        self.do_translation_x = self.do_translation_y = 1.0
        if type(self.do_translation) == list:
            self.do_translation_x = self.do_translation_y = 0
            if 'x' in self.do_translation:
                self.do_translation_x = 1.0
            if 'y' in self.do_translation:
                self.do_translation_y = 1.0
            self.do_translation = True


        self.touches = []

        self.transform_gl = identity_matrix().T.tolist()  #openGL matrix
        self.transform = identity_matrix()
        self.transform_inv = identity_matrix()
        self._current_transform = identity_matrix()

        #inital transformation
        tx,ty = kwargs['translation']
        trans = translation_matrix( (tx, ty, 0) )
        trans = numpy.dot(trans, scale_matrix(kwargs['scale']))
        trans = numpy.dot(trans, rotation_matrix( -kwargs['rotation'], (0, 0, 1)))
        self._prior_transform = trans
        self.update_matrices()


    def _get_transform_mat(self):
        return self.transform
    def _set_transform_mat(self, x):
        self.reset_transformation_origin()
        self._prior_transform = x
        self.update_matrices()
    transform_mat = property(
        _get_transform_mat,
        _set_transform_mat,
        doc='Get/Set transformation matrix (numpy matrix)')


    def to_parent(self, x, y):
        p = numpy.dot(self.transform, (x,y,0,1))
        return (p[0],p[1])

    def to_local(self, x, y):
        p = numpy.dot(self.transform_inv, (x,y,0,1))
        return (p[0],p[1])


    def collide_point(self, x, y):
        if not self.visible:
            return False
        local_coords = self.to_local(x, y)
        if local_coords[0] > 0 and local_coords[0] < self.width \
           and local_coords[1] > 0 and local_coords[1] < self.height:
            return True
        else:
            return False


    def reset_transformation_origin(self):
        for t in self.touches:
            t.userdata['transform_origin'] = (t.x,t.y)

        self._prior_transform = numpy.dot(self._current_transform, self._prior_transform)
        self._current_transform = identity_matrix()
        self.update_matrices()


    def update_matrices(self):
        #compute the OpenGL matrix
        trans = numpy.dot( self._current_transform, self._prior_transform)
        if not is_same_transform(trans, self.transform):
            self.transform = trans
            self.transform_inv = inverse_matrix(self.transform)
            self.transform_gl = self.transform.T.tolist() #for openGL
            self.dispatch_event('on_transform')

    def update_transformation(self):

        #in teh case of one touch, we really just have to translate
        if len(self.touches) == 1:
            dx = self.touches[0].x - self.touches[0].userdata['transform_origin'][0]
            dy = self.touches[0].y - self.touches[0].userdata['transform_origin'][1]
            self._current_transform = translation_matrix((dx,dy,0))
            self.update_matrices()
            return

        #two or more touches...lets do some math!
        """
        heres an attempt at an exmplanation of the math

        we are given two sets of points P1,P2 and P1',P2' (before and after)
        now we are trying to compute the transformation that takes both
        P1->P1' and P2->P2'.  To do this, we have to rely on teh fact that
        teh affine transformation we want is conformal.

        because we know we want a 2D conformal transformation (no shearing, stretching etc)
        we can state the following:

            P1' = M*P1 + t
            P2' = M*P2 + t

            where:
            M is a 2x2 matrix that describes the rotation and scale of the transformation
            t is a 2X1 vector that descrobes the translation of the transformation

        becasue this is a conformal affine transformation (only rotation and scale in M)
        we also know that we can write M as follows:
            |  a  b |
            | -b  a |

            where:
            a = scale * cos(angle)
            b = scale * sin(angle)

        given this and the two equations above, we can rewrite as a system of linear equations:
            x1' =  a*x1 + b*y1 + cx
            y1' = -b*x1 + a*y1 + cy
            x2' =  a*x2 + b*y2 + cx
            y2' = -b*x1 + a*y2 + cy

        the easiest way to solve this is to construct a matrix that takes (a,b,tx,ty)-> (x1',y1',x2',y2')
        and then take its inverse so we can computer a,b,tx,ad ty from the known parameters

           v    =          T         *    x

         |x1'|     | x1  y1  1  0 |      | a|  (notice how multiplying this gives teh 4 equations above)
         |y1'|  =  | y1 -x1  0  1 |  *   | b|
         |x2'|     | x2  y2  1  0 |      |tx|
         |y2'|     | y2 -x2  0  1 |      |ty|

        Now we can easily compute x by multiplying both sides by the inverse of T

            inv(T) * v = inv(T)*T *x  (everything but x cancels out on the right)

        once we have a,b,tx, and ty, we just have to extract teh angle and scale

            angle = artan(b/a)     (based on teh definitionsof a and b above)
            scale = a/cos(angle)


        """

        #old coordinates
        x1 = self.touches[0].userdata['transform_origin'][0]
        y1 = self.touches[0].userdata['transform_origin'][1]
        x2 = self.touches[1].userdata['transform_origin'][0]
        y2 = self.touches[1].userdata['transform_origin'][1]

        #new coordinates
        v = (self.touches[0].x, self.touches[0].y,
             self.touches[1].x, self.touches[1].y )

        #transformation matrix, use 64bit precision
        T = numpy.array(
                ((x1,  y1, 1.0, 0.0),
                 (y1, -x1, 0.0, 1.0),
                 (x2,  y2, 1.0, 0.0),
                 (y2, -x2, 0.0, 1.0)), dtype=numpy.float64 )

        #compute the conformal parameters
        x = numpy.dot(inverse_matrix(T), v)

        a  = x[0]
        b  = x[1]
        tx = x[2] * self.do_translation_x
        ty = x[3] * self.do_translation_y

        angle = atan(b/a)
        scale = a/cos(angle)

        #concatenate transformations based on whther tehy are tunred on/off
        trans = translation_matrix( (tx, ty, 0) )
        if self.do_scale:
            trans = numpy.dot( trans, scale_matrix(scale))
        if self.do_rotation:
            trans = numpy.dot(trans, rotation_matrix( -angle, (0, 0, 1)))

        #update tranformations
        self._current_transform = trans
        self.update_matrices()


    def on_transform(self, *largs):
        pass


    def on_touch_down(self, touch):
        x, y = touch.x, touch.y

        # if the touch isnt on the widget we do nothing
        if not self.collide_point(x, y):
            return False

        # let the child widgets handle the event if they want
        touch.push()
        touch.x, touch.y = self.to_local(x, y)
        if super(MTScatter, self).on_touch_down(touch):
            touch.pop()
            return True
        touch.pop()

        #grab the touch so we get all it later move events for sure
        touch.grab(self)
        self.touches.append(touch)
        self.reset_transformation_origin()

        return True


    def on_touch_move(self, touch):
        x, y = touch.x, touch.y

        # let the child widgets handle the event if they want
        if self.collide_point(x, y) and not touch.grab_current == self:
            touch.push()
            touch.x, touch.y = self.to_local(x, y)
            if super(MTScatter, self).on_touch_move(touch):
                touch.pop()
                return True
            touch.pop()

        # rotate/scale/translate
        if touch in self.touches and touch.grab_current == self:
            self.update_transformation()

        # stop porpagating if its within our bounds
        if self.collide_point(x, y):
            return True


    def on_touch_up(self, touch):
        x, y = touch.x, touch.y

        # if the touch isnt on the widget we do nothing
        if not touch.grab_state:
            touch.push()
            touch.x, touch.y = self.to_local(x, y)
            if super(MTScatter, self).on_touch_up(touch):
                touch.pop()
                return True
            touch.pop()

        # remove it from our saved touches
        if touch in self.touches and touch.grab_state:
            touch.ungrab(self)
            self.touches.remove(touch)
            self.reset_transformation_origin()

        # stop porpagating if its within our bounds
        if self.collide_point(x, y):
            return True


    def on_draw(self):
        if not self.visible:
            return
        with gx_matrix:
            glMultMatrixf(self.transform_gl)
            super(MTScatter, self).on_draw()






runTouchApp(MTScatter(style={'draw-background':1}))



