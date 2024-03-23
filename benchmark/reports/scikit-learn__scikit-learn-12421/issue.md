# scikit-learn__scikit-learn-12421

* repo: scikit-learn/scikit-learn
* base_commit: `013d295a13721ffade7ac321437c6d4458a64c7d`

## Problem statement

OPTICS: self.core_distances_ inconsistent with documentation&R implementation
In the doc, we state that ``Points which will never be core have a distance of inf.``, but it's not the case.
Result from scikit-learn:
```
import numpy as np
from sklearn.cluster import OPTICS
X = np.array([-5, -2, -4.8, -1.8, -5.2, -2.2, 100, 200, 4, 2, 3.8, 1.8, 4.2, 2.2])
X = X.reshape(-1, 2)
clust = OPTICS(min_samples=3, max_bound=1)
clust.fit(X)
clust.core_distances_
```
```
array([  0.28284271,   0.56568542,   0.56568542, 220.04544985, 
         0.28284271,   0.56568542,   0.56568542])
```
Result from R:
```
x <- matrix(c(-5, -2, -4.8, -1.8, -5.2, -2.2, 100, 200,
              4, 2, 3.8, 1.8, 4.2, 2.2), ncol=2, byrow=TRUE)
result <- optics(x, eps=1, minPts=3)
result$coredist
```
```
[1] 0.2828427 0.5656854 0.5656854       Inf 0.2828427 0.5656854 0.5656854
```


## Patch

```diff
diff --git a/sklearn/cluster/optics_.py b/sklearn/cluster/optics_.py
--- a/sklearn/cluster/optics_.py
+++ b/sklearn/cluster/optics_.py
@@ -39,9 +39,8 @@ def optics(X, min_samples=5, max_eps=np.inf, metric='minkowski',
     This implementation deviates from the original OPTICS by first performing
     k-nearest-neighborhood searches on all points to identify core sizes, then
     computing only the distances to unprocessed points when constructing the
-    cluster order. It also does not employ a heap to manage the expansion
-    candiates, but rather uses numpy masked arrays. This can be potentially
-    slower with some parameters (at the benefit from using fast numpy code).
+    cluster order. Note that we do not employ a heap to manage the expansion
+    candidates, so the time complexity will be O(n^2).
 
     Read more in the :ref:`User Guide <optics>`.
 
@@ -199,7 +198,8 @@ class OPTICS(BaseEstimator, ClusterMixin):
     This implementation deviates from the original OPTICS by first performing
     k-nearest-neighborhood searches on all points to identify core sizes, then
     computing only the distances to unprocessed points when constructing the
-    cluster order.
+    cluster order. Note that we do not employ a heap to manage the expansion
+    candidates, so the time complexity will be O(n^2).
 
     Read more in the :ref:`User Guide <optics>`.
 
@@ -430,7 +430,11 @@ def fit(self, X, y=None):
                                 n_jobs=self.n_jobs)
 
         nbrs.fit(X)
+        # Here we first do a kNN query for each point, this differs from
+        # the original OPTICS that only used epsilon range queries.
         self.core_distances_ = self._compute_core_distances_(X, nbrs)
+        # OPTICS puts an upper limit on these, use inf for undefined.
+        self.core_distances_[self.core_distances_ > self.max_eps] = np.inf
         self.ordering_ = self._calculate_optics_order(X, nbrs)
 
         indices_, self.labels_ = _extract_optics(self.ordering_,
@@ -445,7 +449,6 @@ def fit(self, X, y=None):
         return self
 
     # OPTICS helper functions
-
     def _compute_core_distances_(self, X, neighbors, working_memory=None):
         """Compute the k-th nearest neighbor of each sample
 
@@ -485,37 +488,38 @@ def _compute_core_distances_(self, X, neighbors, working_memory=None):
     def _calculate_optics_order(self, X, nbrs):
         # Main OPTICS loop. Not parallelizable. The order that entries are
         # written to the 'ordering_' list is important!
+        # Note that this implementation is O(n^2) theoretically, but
+        # supposedly with very low constant factors.
         processed = np.zeros(X.shape[0], dtype=bool)
         ordering = np.zeros(X.shape[0], dtype=int)
-        ordering_idx = 0
-        for point in range(X.shape[0]):
-            if processed[point]:
-                continue
-            if self.core_distances_[point] <= self.max_eps:
-                while not processed[point]:
-                    processed[point] = True
-                    ordering[ordering_idx] = point
-                    ordering_idx += 1
-                    point = self._set_reach_dist(point, processed, X, nbrs)
-            else:  # For very noisy points
-                ordering[ordering_idx] = point
-                ordering_idx += 1
-                processed[point] = True
+        for ordering_idx in range(X.shape[0]):
+            # Choose next based on smallest reachability distance
+            # (And prefer smaller ids on ties, possibly np.inf!)
+            index = np.where(processed == 0)[0]
+            point = index[np.argmin(self.reachability_[index])]
+
+            processed[point] = True
+            ordering[ordering_idx] = point
+            if self.core_distances_[point] != np.inf:
+                self._set_reach_dist(point, processed, X, nbrs)
         return ordering
 
     def _set_reach_dist(self, point_index, processed, X, nbrs):
         P = X[point_index:point_index + 1]
+        # Assume that radius_neighbors is faster without distances
+        # and we don't need all distances, nevertheless, this means
+        # we may be doing some work twice.
         indices = nbrs.radius_neighbors(P, radius=self.max_eps,
                                         return_distance=False)[0]
 
         # Getting indices of neighbors that have not been processed
         unproc = np.compress((~np.take(processed, indices)).ravel(),
                              indices, axis=0)
-        # Keep n_jobs = 1 in the following lines...please
+        # Neighbors of current point are already processed.
         if not unproc.size:
-            # Everything is already processed. Return to main loop
-            return point_index
+            return
 
+        # Only compute distances to unprocessed neighbors:
         if self.metric == 'precomputed':
             dists = X[point_index, unproc]
         else:
@@ -527,12 +531,6 @@ def _set_reach_dist(self, point_index, processed, X, nbrs):
         self.reachability_[unproc[improved]] = rdists[improved]
         self.predecessor_[unproc[improved]] = point_index
 
-        # Choose next based on smallest reachability distance
-        # (And prefer smaller ids on ties).
-        # All unprocessed points qualify, not just new neighbors ("unproc")
-        return (np.ma.array(self.reachability_, mask=processed)
-                .argmin(fill_value=np.inf))
-
     def extract_dbscan(self, eps):
         """Performs DBSCAN extraction for an arbitrary epsilon.
 

```

## Test Patch

```diff
diff --git a/sklearn/cluster/tests/test_optics.py b/sklearn/cluster/tests/test_optics.py
--- a/sklearn/cluster/tests/test_optics.py
+++ b/sklearn/cluster/tests/test_optics.py
@@ -22,7 +22,7 @@
 
 
 rng = np.random.RandomState(0)
-n_points_per_cluster = 50
+n_points_per_cluster = 10
 C1 = [-5, -2] + .8 * rng.randn(n_points_per_cluster, 2)
 C2 = [4, -1] + .1 * rng.randn(n_points_per_cluster, 2)
 C3 = [1, -2] + .2 * rng.randn(n_points_per_cluster, 2)
@@ -155,16 +155,10 @@ def test_dbscan_optics_parity(eps, min_samples):
     assert percent_mismatch <= 0.05
 
 
-def test_auto_extract_hier():
-    # Tests auto extraction gets correct # of clusters with varying density
-    clust = OPTICS(min_samples=9).fit(X)
-    assert_equal(len(set(clust.labels_)), 6)
-
-
 # try arbitrary minimum sizes
 @pytest.mark.parametrize('min_cluster_size', range(2, X.shape[0] // 10, 23))
 def test_min_cluster_size(min_cluster_size):
-    redX = X[::10]  # reduce for speed
+    redX = X[::2]  # reduce for speed
     clust = OPTICS(min_samples=9, min_cluster_size=min_cluster_size).fit(redX)
     cluster_sizes = np.bincount(clust.labels_[clust.labels_ != -1])
     if cluster_sizes.size:
@@ -215,171 +209,100 @@ def test_cluster_sigmin_pruning(reach, n_child, members):
     assert_array_equal(members, root.children[0].points)
 
 
+def test_processing_order():
+    # Ensure that we consider all unprocessed points,
+    # not only direct neighbors. when picking the next point.
+    Y = [[0], [10], [-10], [25]]
+    clust = OPTICS(min_samples=3, max_eps=15).fit(Y)
+    assert_array_equal(clust.reachability_, [np.inf, 10, 10, 15])
+    assert_array_equal(clust.core_distances_, [10, 15, np.inf, np.inf])
+    assert_array_equal(clust.ordering_, [0, 1, 2, 3])
+
+
 def test_compare_to_ELKI():
     # Expected values, computed with (future) ELKI 0.7.5 using:
     # java -jar elki.jar cli -dbc.in csv -dbc.filter FixedDBIDsFilter
     #   -algorithm clustering.optics.OPTICSHeap -optics.minpts 5
     # where the FixedDBIDsFilter gives 0-indexed ids.
-    r = [np.inf, 0.7865694338710508, 0.4373157299595305, 0.4121908069391695,
-         0.302907091394212, 0.20815674060999778, 0.20815674060999778,
-         0.15190193459676368, 0.15190193459676368, 0.28229645104833345,
-         0.302907091394212, 0.30507239477026865, 0.30820580778767087,
-         0.3289019667317037, 0.3458462228589966, 0.3458462228589966,
-         0.2931114364132193, 0.2931114364132193, 0.2562790168458507,
-         0.23654635530592025, 0.37903448688824876, 0.3920764620583683,
-         0.4121908069391695, 0.4364542226186831, 0.45523658462146793,
-         0.458757846268185, 0.458757846268185, 0.4752907412198826,
-         0.42350366820623375, 0.42350366820623375, 0.42350366820623375,
-         0.47758738570352993, 0.47758738570352993, 0.4776963110272057,
-         0.5272079288923731, 0.5591861752070968, 0.5592057084987357,
-         0.5609913790596295, 0.5909117211348757, 0.5940470220777727,
-         0.5940470220777727, 0.6861627576116127, 0.687795873252133,
-         0.7538541412862811, 0.7865694338710508, 0.8038180561910464,
-         0.8038180561910464, 0.8242451615289921, 0.8548361202185057,
-         0.8790098789921685, 2.9281214555815764, 1.3256656984284734,
-         0.19590944671099267, 0.1339924636672767, 0.1137384200258616,
-         0.061455005237474075, 0.061455005237474075, 0.061455005237474075,
-         0.045627777293497276, 0.045627777293497276, 0.045627777293497276,
-         0.04900902556283447, 0.061455005237474075, 0.06225461602815799,
-         0.06835750467748272, 0.07882900172724974, 0.07882900172724974,
-         0.07650735397943846, 0.07650735397943846, 0.07650735397943846,
-         0.07650735397943846, 0.07650735397943846, 0.07113275489288699,
-         0.07890196345324527, 0.07052683707634783, 0.07052683707634783,
-         0.07052683707634783, 0.08284027053523288, 0.08725436842020087,
-         0.08725436842020087, 0.09010229261951723, 0.09128578974358925,
-         0.09154172670176584, 0.0968576383038391, 0.12007572768323092,
-         0.12024155806196564, 0.12141990481584404, 0.1339924636672767,
-         0.13694322786307633, 0.14275793459246572, 0.15093125027309579,
-         0.17927454395170142, 0.18151803569400365, 0.1906028449191095,
-         0.1906028449191095, 0.19604486784973194, 0.2096539172540186,
-         0.2096539172540186, 0.21614333983312325, 0.22036454909290296,
-         0.23610322103910933, 0.26028003932256766, 0.2607126030060721,
-         0.2891824876072483, 0.3258089271514364, 0.35968687619960743,
-         0.4512973330510512, 0.4746141313843085, 0.5958585488429471,
-         0.6468718886525733, 0.6878453052524358, 0.6911582799500199,
-         0.7172169499815705, 0.7209874999572031, 0.6326884657912096,
-         0.5755681293026617, 0.5755681293026617, 0.5755681293026617,
-         0.6015042225447333, 0.6756244556376542, 0.4722384908959966,
-         0.08775739179493615, 0.06665303472021758, 0.056308477780164796,
-         0.056308477780164796, 0.05507767260835565, 0.05368146914586802,
-         0.05163427719303039, 0.05163427719303039, 0.05163427719303039,
-         0.04918757627098621, 0.04918757627098621, 0.05368146914586802,
-         0.05473720349424546, 0.05473720349424546, 0.048442038421760626,
-         0.048442038421760626, 0.04598840269934622, 0.03984301937835033,
-         0.04598840269934622, 0.04598840269934622, 0.04303884892957088,
-         0.04303884892957088, 0.04303884892957088, 0.0431802780806032,
-         0.0520412490141781, 0.056308477780164796, 0.05080724020124642,
-         0.05080724020124642, 0.05080724020124642, 0.06385565101399236,
-         0.05840878369200427, 0.0474472391259039, 0.0474472391259039,
-         0.04232512684465669, 0.04232512684465669, 0.04232512684465669,
-         0.0474472391259039, 0.051802632822946656, 0.051802632822946656,
-         0.05316405104684577, 0.05316405104684577, 0.05840878369200427,
-         0.06385565101399236, 0.08025248922898705, 0.08775739179493615,
-         0.08993337040710143, 0.08993337040710143, 0.08993337040710143,
-         0.08993337040710143, 0.297457175321605, 0.29763608186278934,
-         0.3415255849656254, 0.34713336941105105, 0.44108940848708167,
-         0.35942962652965604, 0.35942962652965604, 0.33609522256535296,
-         0.5008111387107295, 0.5333587622018111, 0.6223243743872802,
-         0.6793840035409552, 0.7445032492109848, 0.7445032492109848,
-         0.6556432627279256, 0.6556432627279256, 0.6556432627279256,
-         0.8196566935960162, 0.8724089149982351, 0.9352758042365477,
-         0.9352758042365477, 1.0581847953137133, 1.0684332509194163,
-         1.0887817699873303, 1.2552604310322708, 1.3993856001769436,
-         1.4869615658197606, 1.6588098267326852, 1.679969559453028,
-         1.679969559453028, 1.6860509219163458, 1.6860509219163458,
-         1.1465697826627317, 0.992866533434785, 0.7691908270707519,
-         0.578131499171622, 0.578131499171622, 0.578131499171622,
-         0.5754243919945694, 0.8416199360035114, 0.8722493727270406,
-         0.9156549976203665, 0.9156549976203665, 0.7472322844356064,
-         0.715219324518981, 0.715219324518981, 0.715219324518981,
-         0.7472322844356064, 0.820988298336316, 0.908958489674247,
-         0.9234036745782839, 0.9519521817942455, 0.992866533434785,
-         0.992866533434785, 0.9995692674695029, 1.0727415198904493,
-         1.1395519941203158, 1.1395519941203158, 1.1741737271442092,
-         1.212860115632712, 0.8724097897372123, 0.8724097897372123,
-         0.8724097897372123, 1.2439272570611581, 1.2439272570611581,
-         1.3524538390109015, 1.3524538390109015, 1.2982303284415664,
-         1.3610655849680207, 1.3802783392089437, 1.3802783392089437,
-         1.4540636953090629, 1.5879329500533819, 1.5909193228826986,
-         1.72931779186001, 1.9619075944592093, 2.1994355761906257,
-         2.2508672067362165, 2.274436122235927, 2.417635732260135,
-         3.014235905390584, 0.30616929141177107, 0.16449675872754976,
-         0.09071681523805683, 0.09071681523805683, 0.09071681523805683,
-         0.08727060912039632, 0.09151721189581336, 0.12277953408786725,
-         0.14285575406641507, 0.16449675872754976, 0.16321992344119793,
-         0.1330971730344373, 0.11429891993167259, 0.11429891993167259,
-         0.11429891993167259, 0.11429891993167259, 0.11429891993167259,
-         0.0945498340011516, 0.11410457435712089, 0.1196414019798306,
-         0.12925682285016715, 0.12925682285016715, 0.12925682285016715,
-         0.12864887158869853, 0.12864887158869853, 0.12864887158869853,
-         0.13369634918690246, 0.14330826543275352, 0.14877705862323184,
-         0.15203263952428328, 0.15696350160889708, 0.1585326700393211,
-         0.1585326700393211, 0.16034306786654595, 0.16034306786654595,
-         0.15053328296567992, 0.16396729418886688, 0.16763548009617293,
-         0.1732029325454474, 0.21163390061029352, 0.21497664171864372,
-         0.22125889949299, 0.240251070192081, 0.240251070192081,
-         0.2413620965310808, 0.26319419022234064, 0.26319419022234064,
-         0.27989712380504483, 0.2909782800714374]
-    o = [0, 3, 6, 7, 15, 4, 27, 28, 49, 17, 35, 47, 46, 39, 13, 19,
-         22, 29, 30, 38, 34, 32, 43, 8, 25, 9, 37, 23, 33, 40, 44, 11, 36, 5,
-         45, 48, 41, 26, 24, 20, 31, 2, 16, 10, 18, 14, 42, 12, 1, 21, 234,
-         132, 112, 115, 107, 110, 120, 114, 100, 131, 137, 145, 130, 121, 134,
-         116, 149, 108, 111, 113, 142, 148, 119, 104, 126, 133, 138, 127, 101,
-         105, 103, 106, 125, 140, 123, 147, 144, 129, 141, 117, 143, 136, 128,
-         122, 124, 102, 109, 249, 146, 118, 135, 245, 139, 224, 241, 217, 202,
-         248, 233, 214, 236, 211, 206, 231, 212, 221, 229, 244, 208, 226, 83,
-         76, 53, 77, 88, 62, 66, 65, 89, 93, 79, 95, 74, 70, 82, 51, 73, 87,
-         67, 94, 56, 52, 63, 80, 75, 57, 96, 60, 69, 90, 86, 58, 68, 81, 64,
-         84, 85, 97, 59, 98, 61, 71, 78, 92, 50, 91, 55, 54, 72, 99, 210, 201,
-         216, 239, 203, 218, 219, 222, 240, 294, 243, 246, 204, 220, 200, 215,
-         230, 225, 205, 207, 237, 223, 235, 209, 228, 238, 227, 285, 232, 256,
-         281, 270, 260, 252, 272, 268, 292, 298, 269, 275, 257, 250, 284, 283,
-         286, 295, 297, 293, 289, 258, 299, 282, 262, 296, 287, 267, 255, 263,
-         288, 276, 251, 266, 274, 271, 277, 261, 279, 290, 253, 254, 291, 259,
-         280, 278, 273, 247, 265, 242, 264, 213, 199, 174, 154, 152, 180, 186,
-         195, 170, 181, 176, 187, 173, 157, 159, 158, 172, 182, 183, 151, 197,
-         177, 160, 156, 171, 175, 184, 193, 161, 179, 196, 185, 192, 165, 166,
-         164, 189, 155, 162, 188, 153, 178, 169, 194, 150, 163, 198, 190, 191,
-         168, 167]
-    p = [-1, 0, 3, 6, 7, 15, 15, 27, 27, 4, 7, 49, 47, 4, 39, 39,
-         19, 19, 29, 30, 30, 13, 6, 43, 34, 32, 32, 25, 23, 23, 23, 3, 11, 46,
-         46, 45, 9, 38, 33, 26, 26, 8, 20, 33, 0, 18, 18, 2, 18, 44, 0, 234,
-         132, 112, 115, 107, 107, 120, 114, 114, 114, 114, 107, 100, 100, 134,
-         134, 149, 149, 108, 108, 108, 148, 113, 104, 104, 104, 142, 127, 127,
-         126, 138, 126, 148, 127, 148, 127, 112, 147, 116, 117, 101, 145, 128,
-         128, 122, 136, 136, 249, 102, 102, 118, 143, 146, 245, 123, 139, 241,
-         241, 217, 248, 202, 248, 224, 231, 212, 212, 212, 229, 229, 226, 83,
-         76, 53, 53, 88, 62, 66, 66, 66, 93, 93, 79, 93, 70, 82, 82, 73, 87,
-         73, 94, 56, 56, 56, 63, 67, 53, 96, 96, 96, 69, 86, 58, 58, 81, 81,
-         81, 58, 64, 64, 59, 59, 86, 69, 78, 83, 84, 55, 55, 55, 72, 50, 201,
-         210, 216, 203, 203, 219, 54, 240, 239, 240, 236, 236, 220, 220, 220,
-         217, 139, 243, 243, 204, 211, 246, 215, 223, 294, 209, 227, 227, 209,
-         281, 270, 260, 252, 272, 272, 272, 298, 269, 298, 275, 275, 284, 283,
-         283, 283, 284, 286, 283, 298, 299, 260, 260, 250, 299, 258, 258, 296,
-         250, 276, 276, 276, 289, 289, 267, 267, 279, 261, 277, 277, 258, 266,
-         290, 209, 207, 290, 228, 278, 228, 290, 199, 174, 154, 154, 154, 186,
-         186, 180, 170, 174, 187, 173, 157, 159, 157, 157, 159, 183, 183, 172,
-         197, 160, 160, 171, 171, 171, 151, 173, 184, 151, 196, 185, 185, 179,
-         179, 189, 177, 165, 175, 162, 164, 181, 169, 169, 181, 178, 178, 178,
-         168]
+    r1 = [np.inf, 1.0574896366427478, 0.7587934993548423, 0.7290174038973836,
+          0.7290174038973836, 0.7290174038973836, 0.6861627576116127,
+          0.7587934993548423, 0.9280118450166668, 1.1748022534146194,
+          3.3355455741292257, 0.49618389254482587, 0.2552805046961355,
+          0.2552805046961355, 0.24944622248445714, 0.24944622248445714,
+          0.24944622248445714, 0.2552805046961355, 0.2552805046961355,
+          0.3086779122185853, 4.163024452756142, 1.623152630340929,
+          0.45315840475822655, 0.25468325192031926, 0.2254004358159971,
+          0.18765711877083036, 0.1821471333893275, 0.1821471333893275,
+          0.18765711877083036, 0.18765711877083036, 0.2240202988740153,
+          1.154337614548715, 1.342604473837069, 1.323308536402633,
+          0.8607514948648837, 0.27219111215810565, 0.13260875220533205,
+          0.13260875220533205, 0.09890587675958984, 0.09890587675958984,
+          0.13548790801634494, 0.1575483940837384, 0.17515137170530226,
+          0.17575920159442388, 0.27219111215810565, 0.6101447895405373,
+          1.3189208094864302, 1.323308536402633, 2.2509184159764577,
+          2.4517810628594527, 3.675977064404973, 3.8264795626020365,
+          2.9130735341510614, 2.9130735341510614, 2.9130735341510614,
+          2.9130735341510614, 2.8459300127258036, 2.8459300127258036,
+          2.8459300127258036, 3.0321982337972537]
+    o1 = [0, 3, 6, 4, 7, 8, 2, 9, 5, 1, 31, 30, 32, 34, 33, 38, 39, 35, 37, 36,
+          44, 21, 23, 24, 22, 25, 27, 29, 26, 28, 20, 40, 45, 46, 10, 15, 11,
+          13, 17, 19, 18, 12, 16, 14, 47, 49, 43, 48, 42, 41, 53, 57, 51, 52,
+          56, 59, 54, 55, 58, 50]
+    p1 = [-1, 0, 3, 6, 6, 6, 8, 3, 7, 5, 1, 31, 30, 30, 34, 34, 34, 32, 32, 37,
+          36, 44, 21, 23, 24, 22, 25, 25, 22, 22, 22, 21, 40, 45, 46, 10, 15,
+          15, 13, 13, 15, 11, 19, 15, 10, 47, 12, 45, 14, 43, 42, 53, 57, 57,
+          57, 57, 59, 59, 59, 58]
 
     # Tests against known extraction array
     # Does NOT work with metric='euclidean', because sklearn euclidean has
     # worse numeric precision. 'minkowski' is slower but more accurate.
-    clust = OPTICS(min_samples=5).fit(X)
+    clust1 = OPTICS(min_samples=5).fit(X)
 
-    assert_array_equal(clust.ordering_, np.array(o))
-    assert_array_equal(clust.predecessor_[clust.ordering_], np.array(p))
-    assert_allclose(clust.reachability_[clust.ordering_], np.array(r))
+    assert_array_equal(clust1.ordering_, np.array(o1))
+    assert_array_equal(clust1.predecessor_[clust1.ordering_], np.array(p1))
+    assert_allclose(clust1.reachability_[clust1.ordering_], np.array(r1))
     # ELKI currently does not print the core distances (which are not used much
     # in literature, but we can at least ensure to have this consistency:
-    for i in clust.ordering_[1:]:
-        assert (clust.reachability_[i] >=
-                clust.core_distances_[clust.predecessor_[i]])
+    for i in clust1.ordering_[1:]:
+        assert (clust1.reachability_[i] >=
+                clust1.core_distances_[clust1.predecessor_[i]])
+
+    # Expected values, computed with (future) ELKI 0.7.5 using
+    r2 = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
+          np.inf, np.inf, np.inf, 0.27219111215810565, 0.13260875220533205,
+          0.13260875220533205, 0.09890587675958984, 0.09890587675958984,
+          0.13548790801634494, 0.1575483940837384, 0.17515137170530226,
+          0.17575920159442388, 0.27219111215810565, 0.4928068613197889,
+          np.inf, 0.2666183922512113, 0.18765711877083036, 0.1821471333893275,
+          0.1821471333893275, 0.1821471333893275, 0.18715928772277457,
+          0.18765711877083036, 0.18765711877083036, 0.25468325192031926,
+          np.inf, 0.2552805046961355, 0.2552805046961355, 0.24944622248445714,
+          0.24944622248445714, 0.24944622248445714, 0.2552805046961355,
+          0.2552805046961355, 0.3086779122185853, 0.34466409325984865,
+          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
+          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
+          np.inf, np.inf]
+    o2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 11, 13, 17, 19, 18, 12, 16, 14,
+          47, 46, 20, 22, 25, 23, 27, 29, 24, 26, 28, 21, 30, 32, 34, 33, 38,
+          39, 35, 37, 36, 31, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53,
+          54, 55, 56, 57, 58, 59]
+    p2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 15, 15, 13, 13, 15,
+          11, 19, 15, 10, 47, -1, 20, 22, 25, 25, 25, 25, 22, 22, 23, -1, 30,
+          30, 34, 34, 34, 32, 32, 37, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+          -1, -1, -1, -1, -1, -1, -1, -1, -1]
+    clust2 = OPTICS(min_samples=5, max_eps=0.5).fit(X)
+
+    assert_array_equal(clust2.ordering_, np.array(o2))
+    assert_array_equal(clust2.predecessor_[clust2.ordering_], np.array(p2))
+    assert_allclose(clust2.reachability_[clust2.ordering_], np.array(r2))
+
+    index = np.where(clust1.core_distances_ <= 0.5)[0]
+    assert_allclose(clust1.core_distances_[index],
+                    clust2.core_distances_[index])
 
 
 def test_precomputed_dists():
-    redX = X[::10]
+    redX = X[::2]
     dists = pairwise_distances(redX, metric='euclidean')
     clust1 = OPTICS(min_samples=10, algorithm='brute',
                     metric='precomputed').fit(dists)
@@ -388,13 +311,3 @@ def test_precomputed_dists():
 
     assert_allclose(clust1.reachability_, clust2.reachability_)
     assert_array_equal(clust1.labels_, clust2.labels_)
-
-
-def test_processing_order():
-    """Early dev version of OPTICS would not consider all unprocessed points,
-    but only direct neighbors. This tests against this mistake."""
-    Y = [[0], [10], [-10], [25]]
-    clust = OPTICS(min_samples=3, max_eps=15).fit(Y)
-    assert_array_equal(clust.reachability_, [np.inf, 10, 10, 15])
-    assert_array_equal(clust.core_distances_, [10, 15, 20, 25])
-    assert_array_equal(clust.ordering_, [0, 1, 2, 3])

```
