# sympy__sympy-20131

| **sympy/sympy** | `706007ca2fe279020e099d36dd1db0e33123ac4c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7761 |
| **Any found context length** | 2377 |
| **Avg pos** | 20.0 |
| **Min pos** | 5 |
| **Max pos** | 15 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/physics/vector/point.py b/sympy/physics/vector/point.py
--- a/sympy/physics/vector/point.py
+++ b/sympy/physics/vector/point.py
@@ -1,6 +1,7 @@
 from __future__ import print_function, division
 from .vector import Vector, _check_vector
 from .frame import _check_frame
+from warnings import warn
 
 __all__ = ['Point']
 
@@ -529,24 +530,41 @@ def vel(self, frame):
 
         _check_frame(frame)
         if not (frame in self._vel_dict):
+            valid_neighbor_found = False
+            is_cyclic = False
             visited = []
             queue = [self]
+            candidate_neighbor = []
             while queue: #BFS to find nearest point
                 node = queue.pop(0)
                 if node not in visited:
                     visited.append(node)
                     for neighbor, neighbor_pos in node._pos_dict.items():
+                        if neighbor in visited:
+                            continue
                         try:
                             neighbor_pos.express(frame) #Checks if pos vector is valid
                         except ValueError:
                             continue
+                        if neighbor in queue:
+                            is_cyclic = True
                         try :
                             neighbor_velocity = neighbor._vel_dict[frame] #Checks if point has its vel defined in req frame
                         except KeyError:
                             queue.append(neighbor)
                             continue
-                        self.set_vel(frame, self.pos_from(neighbor).dt(frame) + neighbor_velocity)
-                        return self._vel_dict[frame]
+                        candidate_neighbor.append(neighbor)
+                        if not valid_neighbor_found:
+                            self.set_vel(frame, self.pos_from(neighbor).dt(frame) + neighbor_velocity)
+                            valid_neighbor_found = True
+            if is_cyclic:
+                warn('Kinematic loops are defined among the positions of points. This is likely not desired and may cause errors in your calculations.')
+            if len(candidate_neighbor) > 1:
+                warn('Velocity automatically calculated based on point ' +
+                    candidate_neighbor[0].name + ' but it is also possible from points(s):' +
+                    str(candidate_neighbor[1:]) + '. Velocities from these points are not necessarily the same. This may cause errors in your calculations.')
+            if valid_neighbor_found:
+                return self._vel_dict[frame]
             else:
                 raise ValueError('Velocity of point ' + self.name + ' has not been'
                              ' defined in ReferenceFrame ' + frame.name)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/physics/vector/point.py | 4 | 4 | 5 | 2 | 2377
| sympy/physics/vector/point.py | 532 | 533 | 15 | 2 | 7761


## Problem Statement

```
Warn the user when trees of points or trees of reference frames are not self consistent.
sympy.physics.vector has Point and ReferenceFrame. These can be positioned and oriented relative to objects of their same type, respectively. The user is expected to define relative positions and orientations in a consistent manner and the relationships among the objects should be tree graphs. It would be helpful to warn the user if they set positions and orientations that create incorrect graphs, for example adding a cyclic branch; the graphs should be acyclic. You can also create inconsistencies when it comes to calculating velocities and angular velocities, which is done automatically if possible. Here is a point example:

\`\`\`
N = ReferenceFrame('N')
O = Point('O')
P = Point('P')
Q = Point('Q')
P.set_vel(N, N.x)
Q.set_vel(N, N.y)
O.set_pos(P, 5*N.z)
O.set_pos(Q, 6*N.y)
O.vel(N)
\`\`\`

The velocities of O in N are different depending on if you calculate based on P or Q. It is also impossible to choose between P or Q. Right now, P or Q will be chosen based on which comes first in `_pos_dict.items()`. We should warn the user that this graph is inconsistent when trying to calculate the velocity of O in N. This same thing can happen with ReferenceFrame.

I suspect there will be issues when users have kinematic loops and may want to define the loop of point positions that specify the loop. Currently, that is not supported. If you specify a loop through a succession of set_pos() or locate_new() calls, you get an invalid point tree. Kinematic loops have to be dealt with by adding the algebraic equation and forming DAEs instead.

These warnings need some careful thought. The first step would be to define precisely what graphs are consistent and non consistent, in terms of physics.vector's capabilities and design. Once that is defined, some methods to check for consistencies can be added. There will be inconsistencies related purely to position and orientation as well as inconsistencies related to the automated calculation of velocities.

There is discussion in this PR that is relevant: https://github.com/sympy/sympy/pull/20049

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/physics/vector/frame.py | 786 | 837| 718 | 718 | 10262 | 
| 2 | 1 sympy/physics/vector/frame.py | 690 | 703| 158 | 876 | 10262 | 
| 3 | 1 sympy/physics/vector/frame.py | 728 | 785| 802 | 1678 | 10262 | 
| 4 | 1 sympy/physics/vector/frame.py | 720 | 727| 141 | 1819 | 10262 | 
| **-> 5 <-** | **2 sympy/physics/vector/point.py** | 1 | 68| 558 | 2377 | 14921 | 
| 6 | 3 sympy/physics/vector/__init__.py | 1 | 37| 254 | 2631 | 15175 | 
| 7 | 4 sympy/parsing/autolev/test-examples/ruletest9.py | 1 | 56| 793 | 3424 | 15969 | 
| 8 | 5 sympy/diffgeom/rn.py | 108 | 148| 584 | 4008 | 17786 | 
| 9 | 6 sympy/parsing/autolev/test-examples/pydy-example-repo/non_min_pendulum.py | 1 | 37| 507 | 4515 | 18293 | 
| 10 | 7 sympy/parsing/autolev/test-examples/ruletest3.py | 1 | 38| 567 | 5082 | 18860 | 
| 11 | 8 sympy/parsing/autolev/test-examples/ruletest8.py | 1 | 40| 757 | 5839 | 19760 | 
| 12 | 9 sympy/parsing/autolev/test-examples/ruletest10.py | 1 | 48| 753 | 6592 | 20783 | 
| 13 | 10 sympy/physics/mechanics/__init__.py | 1 | 57| 459 | 7051 | 21243 | 
| 14 | 11 sympy/vector/coordsysrect.py | 1 | 24| 228 | 7279 | 29478 | 
| **-> 15 <-** | **11 sympy/physics/vector/point.py** | 499 | 554| 482 | 7761 | 29478 | 
| 16 | 11 sympy/parsing/autolev/test-examples/ruletest10.py | 49 | 65| 269 | 8030 | 29478 | 
| 17 | 12 sympy/parsing/autolev/test-examples/pydy-example-repo/chaos_pendulum.py | 1 | 49| 739 | 8769 | 30322 | 
| 18 | 12 sympy/physics/vector/frame.py | 79 | 92| 120 | 8889 | 30322 | 
| 19 | 13 sympy/parsing/autolev/test-examples/ruletest7.py | 1 | 36| 671 | 9560 | 30994 | 
| 20 | 13 sympy/physics/vector/frame.py | 94 | 220| 1279 | 10839 | 30994 | 
| 21 | 14 sympy/parsing/autolev/test-examples/pydy-example-repo/double_pendulum.py | 1 | 40| 552 | 11391 | 31546 | 
| 22 | 14 sympy/physics/vector/frame.py | 480 | 688| 2733 | 14124 | 31546 | 
| 23 | 15 sympy/physics/mechanics/system.py | 9 | 207| 1886 | 16010 | 35527 | 
| 24 | 16 sympy/parsing/autolev/test-examples/ruletest5.py | 1 | 34| 811 | 16821 | 36338 | 
| 25 | 17 sympy/parsing/autolev/test-examples/pydy-example-repo/mass_spring_damper.py | 1 | 32| 415 | 17236 | 36753 | 
| 26 | 18 sympy/multipledispatch/conflict.py | 1 | 53| 394 | 17630 | 37282 | 
| 27 | 19 sympy/vector/__init__.py | 1 | 48| 470 | 18100 | 37752 | 
| 28 | 20 sympy/parsing/autolev/test-examples/ruletest4.py | 1 | 21| 260 | 18360 | 38012 | 
| 29 | 20 sympy/physics/vector/frame.py | 60 | 76| 125 | 18485 | 38012 | 
| 30 | 21 sympy/utilities/exceptions.py | 1 | 131| 1294 | 19779 | 39755 | 
| 31 | 22 sympy/physics/mechanics/body.py | 1 | 213| 80 | 19859 | 41481 | 
| 32 | 22 sympy/physics/vector/frame.py | 48 | 58| 136 | 19995 | 41481 | 
| 33 | 22 sympy/utilities/exceptions.py | 133 | 187| 458 | 20453 | 41481 | 
| 34 | 23 sympy/physics/continuum_mechanics/beam.py | 1643 | 1758| 1309 | 21762 | 66494 | 
| 35 | 24 sympy/diffgeom/diffgeom.py | 2037 | 2076| 235 | 21997 | 83096 | 
| 36 | 25 sympy/geometry/point.py | 1 | 36| 250 | 22247 | 92545 | 
| 37 | 26 sympy/parsing/autolev/test-examples/ruletest2.py | 1 | 23| 358 | 22605 | 92904 | 
| 38 | 27 sympy/parsing/autolev/_listener_autolev_antlr.py | 996 | 1089| 1226 | 23831 | 116025 | 
| 39 | 28 sympy/physics/gaussopt.py | 1 | 20| 252 | 24083 | 116277 | 
| 40 | 28 sympy/physics/vector/frame.py | 839 | 924| 829 | 24912 | 116277 | 
| 41 | 29 sympy/integrals/meijerint.py | 1175 | 1212| 760 | 25672 | 140555 | 
| 42 | 30 sympy/physics/mechanics/functions.py | 1 | 47| 325 | 25997 | 146770 | 
| 43 | 30 sympy/physics/vector/frame.py | 962 | 1011| 329 | 26326 | 146770 | 
| 44 | 31 sympy/polys/rootisolation.py | 732 | 786| 647 | 26973 | 167365 | 
| 45 | 32 sympy/physics/vector/functions.py | 274 | 323| 1128 | 28101 | 174228 | 
| 46 | 32 sympy/physics/vector/frame.py | 1 | 46| 307 | 28408 | 174228 | 
| 47 | 32 sympy/vector/coordsysrect.py | 134 | 208| 827 | 29235 | 174228 | 
| 48 | **32 sympy/physics/vector/point.py** | 370 | 397| 177 | 29412 | 174228 | 
| 49 | 32 sympy/integrals/meijerint.py | 1096 | 1133| 808 | 30220 | 174228 | 
| 50 | 33 setup.py | 79 | 168| 815 | 31035 | 177966 | 
| 51 | 34 sympy/diffgeom/__init__.py | 1 | 20| 282 | 31317 | 178248 | 
| 52 | 34 sympy/geometry/point.py | 106 | 176| 613 | 31930 | 178248 | 
| 53 | 34 sympy/integrals/meijerint.py | 1053 | 1095| 741 | 32671 | 178248 | 
| 54 | 34 sympy/physics/vector/frame.py | 705 | 718| 182 | 32853 | 178248 | 
| 55 | 35 sympy/geometry/polygon.py | 1 | 20| 191 | 33044 | 199799 | 


### Hint

```
If we have multiple points defined at a same level , currently automated velocity would choose the point which comes first, but what I suggest is that we calculate all possible velocities of shortest path by and update _vel_dict by a dictionary

p._vel_dict[frame] = { point1 : calculated_velocity, point2 : calc_velocity}

point1 , point2 being two points with velocity defined with required frame at same level.

This could give user a choice to choose between any velocity he requires

If he updates p's velocity using set_vel(frame) then the dictionary is overridden by user defined velocity.
That's not what we'd want. The goal is to calculate the correct velocity or none at all.
> That's not what we'd want. The goal is to calculate the correct velocity or none at all.

Got it
```

## Patch

```diff
diff --git a/sympy/physics/vector/point.py b/sympy/physics/vector/point.py
--- a/sympy/physics/vector/point.py
+++ b/sympy/physics/vector/point.py
@@ -1,6 +1,7 @@
 from __future__ import print_function, division
 from .vector import Vector, _check_vector
 from .frame import _check_frame
+from warnings import warn
 
 __all__ = ['Point']
 
@@ -529,24 +530,41 @@ def vel(self, frame):
 
         _check_frame(frame)
         if not (frame in self._vel_dict):
+            valid_neighbor_found = False
+            is_cyclic = False
             visited = []
             queue = [self]
+            candidate_neighbor = []
             while queue: #BFS to find nearest point
                 node = queue.pop(0)
                 if node not in visited:
                     visited.append(node)
                     for neighbor, neighbor_pos in node._pos_dict.items():
+                        if neighbor in visited:
+                            continue
                         try:
                             neighbor_pos.express(frame) #Checks if pos vector is valid
                         except ValueError:
                             continue
+                        if neighbor in queue:
+                            is_cyclic = True
                         try :
                             neighbor_velocity = neighbor._vel_dict[frame] #Checks if point has its vel defined in req frame
                         except KeyError:
                             queue.append(neighbor)
                             continue
-                        self.set_vel(frame, self.pos_from(neighbor).dt(frame) + neighbor_velocity)
-                        return self._vel_dict[frame]
+                        candidate_neighbor.append(neighbor)
+                        if not valid_neighbor_found:
+                            self.set_vel(frame, self.pos_from(neighbor).dt(frame) + neighbor_velocity)
+                            valid_neighbor_found = True
+            if is_cyclic:
+                warn('Kinematic loops are defined among the positions of points. This is likely not desired and may cause errors in your calculations.')
+            if len(candidate_neighbor) > 1:
+                warn('Velocity automatically calculated based on point ' +
+                    candidate_neighbor[0].name + ' but it is also possible from points(s):' +
+                    str(candidate_neighbor[1:]) + '. Velocities from these points are not necessarily the same. This may cause errors in your calculations.')
+            if valid_neighbor_found:
+                return self._vel_dict[frame]
             else:
                 raise ValueError('Velocity of point ' + self.name + ' has not been'
                              ' defined in ReferenceFrame ' + frame.name)

```

## Test Patch

```diff
diff --git a/sympy/physics/vector/tests/test_point.py b/sympy/physics/vector/tests/test_point.py
--- a/sympy/physics/vector/tests/test_point.py
+++ b/sympy/physics/vector/tests/test_point.py
@@ -1,6 +1,6 @@
 from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
-from sympy.testing.pytest import raises
-
+from sympy.testing.pytest import raises, ignore_warnings
+import warnings
 
 def test_point_v1pt_theorys():
     q, q2 = dynamicsymbols('q q2')
@@ -216,7 +216,10 @@ def test_auto_point_vel_shortest_path():
     O1 = Point('O1')
     O1.set_pos(O, q2 * B.z)
     P4.set_pos(O1, q1 * B.x + q2 * B.z)
-    assert P4.vel(B) == q1.diff(t) * B.x + u2 * B.y + 2 * q2.diff(t) * B.z
+    with warnings.catch_warnings(): #There are two possible paths in this point tree, thus a warning is raised
+        warnings.simplefilter('error')
+        with ignore_warnings(UserWarning):
+            assert P4.vel(B) == q1.diff(t) * B.x + u2 * B.y + 2 * q2.diff(t) * B.z
 
 def test_auto_point_vel_connected_frames():
     t = dynamicsymbols._t
@@ -230,3 +233,68 @@ def test_auto_point_vel_connected_frames():
     raises(ValueError, lambda: P.vel(N))
     N.orient(B, 'Axis', (q, B.x))
     assert P.vel(N) == (u + q1.diff(t)) * N.x + q2.diff(t) * B.y - q2 * q.diff(t) * B.z
+
+def test_auto_point_vel_multiple_paths_warning_arises():
+    q, u = dynamicsymbols('q u')
+    N = ReferenceFrame('N')
+    O = Point('O')
+    P = Point('P')
+    Q = Point('Q')
+    R = Point('R')
+    P.set_vel(N, u * N.x)
+    Q.set_vel(N, u *N.y)
+    R.set_vel(N, u * N.z)
+    O.set_pos(P, q * N.z)
+    O.set_pos(Q, q * N.y)
+    O.set_pos(R, q * N.x)
+    with warnings.catch_warnings(): #There are two possible paths in this point tree, thus a warning is raised
+        warnings.simplefilter("error")
+        raises(UserWarning ,lambda: O.vel(N))
+
+def test_auto_vel_cyclic_warning_arises():
+    P = Point('P')
+    P1 = Point('P1')
+    P2 = Point('P2')
+    P3 = Point('P3')
+    N = ReferenceFrame('N')
+    P.set_vel(N, N.x)
+    P1.set_pos(P, N.x)
+    P2.set_pos(P1, N.y)
+    P3.set_pos(P2, N.z)
+    P1.set_pos(P3, N.x + N.y)
+    with warnings.catch_warnings(): #The path is cyclic at P1, thus a warning is raised
+        warnings.simplefilter("error")
+        raises(UserWarning ,lambda: P2.vel(N))
+
+def test_auto_vel_cyclic_warning_msg():
+    P = Point('P')
+    P1 = Point('P1')
+    P2 = Point('P2')
+    P3 = Point('P3')
+    N = ReferenceFrame('N')
+    P.set_vel(N, N.x)
+    P1.set_pos(P, N.x)
+    P2.set_pos(P1, N.y)
+    P3.set_pos(P2, N.z)
+    P1.set_pos(P3, N.x + N.y)
+    with warnings.catch_warnings(record = True) as w: #The path is cyclic at P1, thus a warning is raised
+        warnings.simplefilter("always")
+        P2.vel(N)
+        assert issubclass(w[-1].category, UserWarning)
+        assert 'Kinematic loops are defined among the positions of points. This is likely not desired and may cause errors in your calculations.' in str(w[-1].message)
+
+def test_auto_vel_multiple_path_warning_msg():
+    N = ReferenceFrame('N')
+    O = Point('O')
+    P = Point('P')
+    Q = Point('Q')
+    P.set_vel(N, N.x)
+    Q.set_vel(N, N.y)
+    O.set_pos(P, N.z)
+    O.set_pos(Q, N.y)
+    with warnings.catch_warnings(record = True) as w: #There are two possible paths in this point tree, thus a warning is raised
+        warnings.simplefilter("always")
+        O.vel(N)
+        assert issubclass(w[-1].category, UserWarning)
+        assert 'Velocity automatically calculated based on point' in str(w[-1].message)
+        assert 'Velocities from these points are not necessarily the same. This may cause errors in your calculations.' in str(w[-1].message)

```


## Code snippets

### 1 - sympy/physics/vector/frame.py:

Start line: 786, End line: 837

```python
class ReferenceFrame(object):

    def orient(self, parent, rot_type, amounts, rot_order=''):
        # ... other code
        dcm_cache_del = []
        for frame in frames:
            if frame in self._dcm_dict:
                dcm_dict_del += [frame]
            dcm_cache_del += [frame]
        for frame in dcm_dict_del:
            del frame._dcm_dict[self]
        for frame in dcm_cache_del:
            del frame._dcm_cache[self]
        # Add the dcm relationship to _dcm_dict
        self._dcm_dict = self._dlist[0] = {}
        self._dcm_dict.update({parent: parent_orient.T})
        parent._dcm_dict.update({self: parent_orient})
        # Also update the dcm cache after resetting it
        self._dcm_cache = {}
        self._dcm_cache.update({parent: parent_orient.T})
        parent._dcm_cache.update({self: parent_orient})
        if rot_type == 'QUATERNION':
            t = dynamicsymbols._t
            q0, q1, q2, q3 = amounts
            q0d = diff(q0, t)
            q1d = diff(q1, t)
            q2d = diff(q2, t)
            q3d = diff(q3, t)
            w1 = 2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1)
            w2 = 2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2)
            w3 = 2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3)
            wvec = Vector([(Matrix([w1, w2, w3]), self)])
        elif rot_type == 'AXIS':
            thetad = (amounts[0]).diff(dynamicsymbols._t)
            wvec = thetad * amounts[1].express(parent).normalize()
        elif rot_type == 'DCM':
            wvec = self._w_diff_dcm(parent)
        else:
            try:
                from sympy.polys.polyerrors import CoercionFailed
                from sympy.physics.vector.functions import kinematic_equations
                q1, q2, q3 = amounts
                u1, u2, u3 = symbols('u1, u2, u3', cls=Dummy)
                templist = kinematic_equations([u1, u2, u3], [q1, q2, q3],
                                               rot_type, rot_order)
                templist = [expand(i) for i in templist]
                td = solve(templist, [u1, u2, u3])
                u1 = expand(td[u1])
                u2 = expand(td[u2])
                u3 = expand(td[u3])
                wvec = u1 * self.x + u2 * self.y + u3 * self.z
            except (CoercionFailed, AssertionError):
                wvec = self._w_diff_dcm(parent)
        self._ang_vel_dict.update({parent: wvec})
        parent._ang_vel_dict.update({self: -wvec})
        self._var_dict = {}
```
### 2 - sympy/physics/vector/frame.py:

Start line: 690, End line: 703

```python
class ReferenceFrame(object):

    def orient(self, parent, rot_type, amounts, rot_order=''):

        from sympy.physics.vector.functions import dynamicsymbols
        _check_frame(parent)

        # Allow passing a rotation matrix manually.
        if rot_type == 'DCM':
            # When rot_type == 'DCM', then amounts must be a Matrix type object
            # (e.g. sympy.matrices.dense.MutableDenseMatrix).
            if not isinstance(amounts, MatrixBase):
                raise TypeError("Amounts must be a sympy Matrix type object.")
        else:
            amounts = list(amounts)
            for i, v in enumerate(amounts):
                if not isinstance(v, Vector):
                    amounts[i] = sympify(v)
        # ... other code
```
### 3 - sympy/physics/vector/frame.py:

Start line: 728, End line: 785

```python
class ReferenceFrame(object):

    def orient(self, parent, rot_type, amounts, rot_order=''):
        # ... other code
        if rot_type == 'AXIS':
            if not rot_order == '':
                raise TypeError('Axis orientation takes no rotation order')
            if not (isinstance(amounts, (list, tuple)) & (len(amounts) == 2)):
                raise TypeError('Amounts are a list or tuple of length 2')
            theta = amounts[0]
            axis = amounts[1]
            axis = _check_vector(axis)
            if not axis.dt(parent) == 0:
                raise ValueError('Axis cannot be time-varying')
            axis = axis.express(parent).normalize()
            axis = axis.args[0][0]
            parent_orient = ((eye(3) - axis * axis.T) * cos(theta) +
                             Matrix([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]]) *
                             sin(theta) + axis * axis.T)
        elif rot_type == 'QUATERNION':
            if not rot_order == '':
                raise TypeError(
                    'Quaternion orientation takes no rotation order')
            if not (isinstance(amounts, (list, tuple)) & (len(amounts) == 4)):
                raise TypeError('Amounts are a list or tuple of length 4')
            q0, q1, q2, q3 = amounts
            parent_orient = (Matrix([[q0**2 + q1**2 - q2**2 - q3**2,
                                      2 * (q1 * q2 - q0 * q3),
                                      2 * (q0 * q2 + q1 * q3)],
                                     [2 * (q1 * q2 + q0 * q3),
                                      q0**2 - q1**2 + q2**2 - q3**2,
                                      2 * (q2 * q3 - q0 * q1)],
                                     [2 * (q1 * q3 - q0 * q2),
                                      2 * (q0 * q1 + q2 * q3),
                                      q0**2 - q1**2 - q2**2 + q3**2]]))
        elif rot_type == 'BODY':
            if not (len(amounts) == 3 & len(rot_order) == 3):
                raise TypeError('Body orientation takes 3 values & 3 orders')
            a1 = int(rot_order[0])
            a2 = int(rot_order[1])
            a3 = int(rot_order[2])
            parent_orient = (_rot(a1, amounts[0]) * _rot(a2, amounts[1]) *
                             _rot(a3, amounts[2]))
        elif rot_type == 'SPACE':
            if not (len(amounts) == 3 & len(rot_order) == 3):
                raise TypeError('Space orientation takes 3 values & 3 orders')
            a1 = int(rot_order[0])
            a2 = int(rot_order[1])
            a3 = int(rot_order[2])
            parent_orient = (_rot(a3, amounts[2]) * _rot(a2, amounts[1]) *
                             _rot(a1, amounts[0]))
        elif rot_type == 'DCM':
            parent_orient = amounts
        else:
            raise NotImplementedError('That is not an implemented rotation')
        # Reset the _dcm_cache of this frame, and remove it from the
        # _dcm_caches of the frames it is linked to. Also remove it from the
        # _dcm_dict of its parent
        frames = self._dcm_cache.keys()
        dcm_dict_del = []
        # ... other code
```
### 4 - sympy/physics/vector/frame.py:

Start line: 720, End line: 727

```python
class ReferenceFrame(object):

    def orient(self, parent, rot_type, amounts, rot_order=''):
        # ... other code

        approved_orders = ('123', '231', '312', '132', '213', '321', '121',
                           '131', '212', '232', '313', '323', '')
        # make sure XYZ => 123 and rot_type is in upper case
        rot_order = translate(str(rot_order), 'XYZxyz', '123123')
        rot_type = rot_type.upper()
        if rot_order not in approved_orders:
            raise TypeError('The supplied order is not an approved type')
        parent_orient = []
        # ... other code
```
### 5 - sympy/physics/vector/point.py:

Start line: 1, End line: 68

```python
from __future__ import print_function, division
from .vector import Vector, _check_vector
from .frame import _check_frame

__all__ = ['Point']


class Point(object):
    """This object represents a point in a dynamic system.

    It stores the: position, velocity, and acceleration of a point.
    The position is a vector defined as the vector distance from a parent
    point to this point.

    Parameters
    ==========

    name : string
        The display name of the Point

    Examples
    ========

    >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> P = Point('P')
    >>> u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    >>> O.set_vel(N, u1 * N.x + u2 * N.y + u3 * N.z)
    >>> O.acc(N)
    u1'*N.x + u2'*N.y + u3'*N.z

    symbols() can be used to create multiple Points in a single step, for example:

    >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import symbols
    >>> N = ReferenceFrame('N')
    >>> u1, u2 = dynamicsymbols('u1 u2')
    >>> A, B = symbols('A B', cls=Point)
    >>> type(A)
    <class 'sympy.physics.vector.point.Point'>
    >>> A.set_vel(N, u1 * N.x + u2 * N.y)
    >>> B.set_vel(N, u2 * N.x + u1 * N.y)
    >>> A.acc(N) - B.acc(N)
    (u1' - u2')*N.x + (-u1' + u2')*N.y

    """

    def __init__(self, name):
        """Initialization of a Point object. """
        self.name = name
        self._pos_dict = {}
        self._vel_dict = {}
        self._acc_dict = {}
        self._pdlist = [self._pos_dict, self._vel_dict, self._acc_dict]

    def __str__(self):
        return self.name

    __repr__ = __str__

    def _check_point(self, other):
        if not isinstance(other, Point):
            raise TypeError('A Point must be supplied')
```
### 6 - sympy/physics/vector/__init__.py:

Start line: 1, End line: 37

```python
__all__ = [
    'CoordinateSym', 'ReferenceFrame',

    'Dyadic',

    'Vector',

    'Point',

    'cross', 'dot', 'express', 'time_derivative', 'outer',
    'kinematic_equations', 'get_motion_params', 'partial_velocity',
    'dynamicsymbols',

    'vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex', 'init_vprinting',

    'curl', 'divergence', 'gradient', 'is_conservative', 'is_solenoidal',
    'scalar_potential', 'scalar_potential_difference',

]
from .frame import CoordinateSym, ReferenceFrame

from .dyadic import Dyadic

from .vector import Vector

from .point import Point

from .functions import (cross, dot, express, time_derivative, outer,
        kinematic_equations, get_motion_params, partial_velocity,
        dynamicsymbols)

from .printing import (vprint, vsstrrepr, vsprint, vpprint, vlatex,
        init_vprinting)

from .fieldfunctions import (curl, divergence, gradient, is_conservative,
        is_solenoidal, scalar_potential, scalar_potential_difference)
```
### 7 - sympy/parsing/autolev/test-examples/ruletest9.py:

Start line: 1, End line: 56

```python
import sympy.physics.mechanics as me
import sympy as sm
import math as m
import numpy as np

frame_n = me.ReferenceFrame('n')
frame_a = me.ReferenceFrame('a')
a=0
d=me.inertia(frame_a, 1, 1, 1)
point_po1 = me.Point('po1')
point_po2 = me.Point('po2')
particle_p1 = me.Particle('p1', me.Point('p1_pt'), sm.Symbol('m'))
particle_p2 = me.Particle('p2', me.Point('p2_pt'), sm.Symbol('m'))
c1, c2, c3 = me.dynamicsymbols('c1 c2 c3')
c1d, c2d, c3d = me.dynamicsymbols('c1 c2 c3', 1)
body_r_cm = me.Point('r_cm')
body_r_cm.set_vel(frame_n, 0)
body_r_f = me.ReferenceFrame('r_f')
body_r = me.RigidBody('r', body_r_cm, body_r_f, sm.symbols('m'), (me.outer(body_r_f.x,body_r_f.x),body_r_cm))
point_po2.set_pos(particle_p1.point, c1*frame_a.x)
v=2*point_po2.pos_from(particle_p1.point)+c2*frame_a.y
frame_a.set_ang_vel(frame_n, c3*frame_a.z)
v=2*frame_a.ang_vel_in(frame_n)+c2*frame_a.y
body_r_f.set_ang_vel(frame_n, c3*frame_a.z)
v=2*body_r_f.ang_vel_in(frame_n)+c2*frame_a.y
frame_a.set_ang_acc(frame_n, (frame_a.ang_vel_in(frame_n)).dt(frame_a))
v=2*frame_a.ang_acc_in(frame_n)+c2*frame_a.y
particle_p1.point.set_vel(frame_a, c1*frame_a.x+c3*frame_a.y)
body_r_cm.set_acc(frame_n, c2*frame_a.y)
v_a = me.cross(body_r_cm.acc(frame_n), particle_p1.point.vel(frame_a))
x_b_c = v_a
x_b_d = 2*x_b_c
a_b_c_d_e=x_b_d*2
a_b_c = 2*c1*c2*c3
a_b_c += 2*c1
a_b_c  =  3*c1
q1, q2, u1, u2 = me.dynamicsymbols('q1 q2 u1 u2')
q1d, q2d, u1d, u2d = me.dynamicsymbols('q1 q2 u1 u2', 1)
x, y = me.dynamicsymbols('x y')
xd, yd = me.dynamicsymbols('x y', 1)
xd2, yd2 = me.dynamicsymbols('x y', 2)
yy = me.dynamicsymbols('yy')
yy = x*xd**2+1
m = sm.Matrix([[0]])
m[0] = 2*x
m = m.row_insert(m.shape[0], sm.Matrix([[0]]))
m[m.shape[0]-1] = 2*y
a = 2*m[0]
m = sm.Matrix([1,2,3,4,5,6,7,8,9]).reshape(3, 3)
m[0,1]=5
a = m[0, 1]*2
force_ro = q1*frame_n.x
torque_a = q2*frame_n.z
force_ro = q1*frame_n.x + q2*frame_n.y
f=force_ro*2
```
### 8 - sympy/diffgeom/rn.py:

Start line: 108, End line: 148

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x, y, z, rho, psi, r, theta, phi = symbols('x y z rho psi r theta phi', cls=Dummy)
    R3_r.connect_to(R3_c, [x, y, z],
                        [sqrt(x**2 + y**2), atan2(y, x), z],
                    inverse=False, fill_in_gaps=False)
    R3_c.connect_to(R3_r, [rho, psi, z],
                        [rho*cos(psi), rho*sin(psi), z],
                    inverse=False, fill_in_gaps=False)
    ## rectangular <-> spherical
    R3_r.connect_to(R3_s, [x, y, z],
                        [sqrt(x**2 + y**2 + z**2), acos(z/
                                sqrt(x**2 + y**2 + z**2)), atan2(y, x)],
                    inverse=False, fill_in_gaps=False)
    R3_s.connect_to(R3_r, [r, theta, phi],
                        [r*sin(theta)*cos(phi), r*sin(
                            theta)*sin(phi), r*cos(theta)],
                    inverse=False, fill_in_gaps=False)
    ## cylindrical <-> spherical
    R3_c.connect_to(R3_s, [rho, psi, z],
                        [sqrt(rho**2 + z**2), acos(z/sqrt(rho**2 + z**2)), psi],
                    inverse=False, fill_in_gaps=False)
    R3_s.connect_to(R3_c, [r, theta, phi],
                        [r*sin(theta), phi, r*cos(theta)],
                    inverse=False, fill_in_gaps=False)

# Defining the basis coordinate functions.
R3_r.x, R3_r.y, R3_r.z = R3_r.coord_functions()
R3_c.rho, R3_c.psi, R3_c.z = R3_c.coord_functions()
R3_s.r, R3_s.theta, R3_s.phi = R3_s.coord_functions()

# Defining the basis vector fields.
R3_r.e_x, R3_r.e_y, R3_r.e_z = R3_r.base_vectors()
R3_c.e_rho, R3_c.e_psi, R3_c.e_z = R3_c.base_vectors()
R3_s.e_r, R3_s.e_theta, R3_s.e_phi = R3_s.base_vectors()

# Defining the basis oneform fields.
R3_r.dx, R3_r.dy, R3_r.dz = R3_r.base_oneforms()
R3_c.drho, R3_c.dpsi, R3_c.dz = R3_c.base_oneforms()
R3_s.dr, R3_s.dtheta, R3_s.dphi = R3_s.base_oneforms()
```
### 9 - sympy/parsing/autolev/test-examples/pydy-example-repo/non_min_pendulum.py:

Start line: 1, End line: 37

```python
import sympy.physics.mechanics as me
import sympy as sm
import math as m
import numpy as np

q1, q2 = me.dynamicsymbols('q1 q2')
q1d, q2d = me.dynamicsymbols('q1 q2', 1)
q1d2, q2d2 = me.dynamicsymbols('q1 q2', 2)
l, m, g = sm.symbols('l m g', real=True)
frame_n = me.ReferenceFrame('n')
point_pn = me.Point('pn')
point_pn.set_vel(frame_n, 0)
theta1 = sm.atan(q2/q1)
frame_a = me.ReferenceFrame('a')
frame_a.orient(frame_n, 'Axis', [theta1, frame_n.z])
particle_p = me.Particle('p', me.Point('p_pt'), sm.Symbol('m'))
particle_p.point.set_pos(point_pn, q1*frame_n.x+q2*frame_n.y)
particle_p.mass = m
particle_p.point.set_vel(frame_n, (point_pn.pos_from(particle_p.point)).dt(frame_n))
f_v = me.dot((particle_p.point.vel(frame_n)).express(frame_a), frame_a.x)
force_p = particle_p.mass*(g*frame_n.x)
dependent = sm.Matrix([[0]])
dependent[0] = f_v
velocity_constraints = [i for i in dependent]
u_q1d = me.dynamicsymbols('u_q1d')
u_q2d = me.dynamicsymbols('u_q2d')
kd_eqs = [q1d-u_q1d, q2d-u_q2d]
forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x))]
kane = me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u_q2d], u_dependent=[u_q1d], kd_eqs = kd_eqs, velocity_constraints = velocity_constraints)
fr, frstar = kane.kanes_equations([particle_p], forceList)
zero = fr+frstar
f_c = point_pn.pos_from(particle_p.point).magnitude()-l
config = sm.Matrix([[0]])
config[0] = f_c
zero = zero.row_insert(zero.shape[0], sm.Matrix([[0]]))
zero[zero.shape[0]-1] = config[0]
```
### 10 - sympy/parsing/autolev/test-examples/ruletest3.py:

Start line: 1, End line: 38

```python
import sympy.physics.mechanics as me
import sympy as sm
import math as m
import numpy as np

frame_a = me.ReferenceFrame('a')
frame_b = me.ReferenceFrame('b')
frame_n = me.ReferenceFrame('n')
x1, x2, x3 = me.dynamicsymbols('x1 x2 x3')
l = sm.symbols('l', real=True)
v1=x1*frame_a.x+x2*frame_a.y+x3*frame_a.z
v2=x1*frame_b.x+x2*frame_b.y+x3*frame_b.z
v3=x1*frame_n.x+x2*frame_n.y+x3*frame_n.z
v=v1+v2+v3
point_c = me.Point('c')
point_d = me.Point('d')
point_po1 = me.Point('po1')
point_po2 = me.Point('po2')
point_po3 = me.Point('po3')
particle_l = me.Particle('l', me.Point('l_pt'), sm.Symbol('m'))
particle_p1 = me.Particle('p1', me.Point('p1_pt'), sm.Symbol('m'))
particle_p2 = me.Particle('p2', me.Point('p2_pt'), sm.Symbol('m'))
particle_p3 = me.Particle('p3', me.Point('p3_pt'), sm.Symbol('m'))
body_s_cm = me.Point('s_cm')
body_s_cm.set_vel(frame_n, 0)
body_s_f = me.ReferenceFrame('s_f')
body_s = me.RigidBody('s', body_s_cm, body_s_f, sm.symbols('m'), (me.outer(body_s_f.x,body_s_f.x),body_s_cm))
body_r1_cm = me.Point('r1_cm')
body_r1_cm.set_vel(frame_n, 0)
body_r1_f = me.ReferenceFrame('r1_f')
body_r1 = me.RigidBody('r1', body_r1_cm, body_r1_f, sm.symbols('m'), (me.outer(body_r1_f.x,body_r1_f.x),body_r1_cm))
body_r2_cm = me.Point('r2_cm')
body_r2_cm.set_vel(frame_n, 0)
body_r2_f = me.ReferenceFrame('r2_f')
body_r2 = me.RigidBody('r2', body_r2_cm, body_r2_f, sm.symbols('m'), (me.outer(body_r2_f.x,body_r2_f.x),body_r2_cm))
v4=x1*body_s_f.x+x2*body_s_f.y+x3*body_s_f.z
body_s_cm.set_pos(point_c, l*frame_n.x)
```
### 15 - sympy/physics/vector/point.py:

Start line: 499, End line: 554

```python
class Point(object):

    def vel(self, frame):
        """The velocity Vector of this Point in the ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which the returned velocity vector will be defined in

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_vel(N, 10 * N.x)
        >>> p1.vel(N)
        10*N.x

        Velocities will be automatically calculated if possible, otherwise a ``ValueError`` will be returned. If it is possible to calculate multiple different velocities from the relative points, the points defined most directly relative to this point will be used. In the case of inconsistent relative positions of points, incorrect velocities may be returned. It is up to the user to define prior relative positions and velocities of points in a self-consistent way.

        >>> p = Point('p')
        >>> q = dynamicsymbols('q')
        >>> p.set_vel(N, 10 * N.x)
        >>> p2 = Point('p2')
        >>> p2.set_pos(p, q*N.x)
        >>> p2.vel(N)
        (Derivative(q(t), t) + 10)*N.x

        """

        _check_frame(frame)
        if not (frame in self._vel_dict):
            visited = []
            queue = [self]
            while queue: #BFS to find nearest point
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    for neighbor, neighbor_pos in node._pos_dict.items():
                        try:
                            neighbor_pos.express(frame) #Checks if pos vector is valid
                        except ValueError:
                            continue
                        try :
                            neighbor_velocity = neighbor._vel_dict[frame] #Checks if point has its vel defined in req frame
                        except KeyError:
                            queue.append(neighbor)
                            continue
                        self.set_vel(frame, self.pos_from(neighbor).dt(frame) + neighbor_velocity)
                        return self._vel_dict[frame]
            else:
                raise ValueError('Velocity of point ' + self.name + ' has not been'
                             ' defined in ReferenceFrame ' + frame.name)

        return self._vel_dict[frame]
```
### 48 - sympy/physics/vector/point.py:

Start line: 370, End line: 397

```python
class Point(object):

    def set_vel(self, frame, value):
        """Sets the velocity Vector of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which this point's velocity is defined
        value : Vector
            The vector value of this point's velocity in the frame

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_vel(N, 10 * N.x)
        >>> p1.vel(N)
        10*N.x

        """

        if value == 0:
            value = Vector(0)
        value = _check_vector(value)
        _check_frame(frame)
        self._vel_dict.update({frame: value})
```
