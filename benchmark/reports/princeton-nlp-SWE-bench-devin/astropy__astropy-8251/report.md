# astropy__astropy-8251

| **astropy/astropy** | `2002221360f4ad75f6b275bbffe4fa68412299b3` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 2692 |
| **Any found context length** | 2692 |
| **Avg pos** | 26.0 |
| **Min pos** | 7 |
| **Max pos** | 36 |
| **Top file pos** | 4 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/units/format/generic.py b/astropy/units/format/generic.py
--- a/astropy/units/format/generic.py
+++ b/astropy/units/format/generic.py
@@ -274,7 +274,9 @@ def p_factor_int(p):
         def p_factor_fits(p):
             '''
             factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN
+                        | UINT power OPEN_PAREN UINT CLOSE_PAREN
                         | UINT power signed_int
+                        | UINT power UINT
                         | UINT SIGN UINT
                         | UINT OPEN_PAREN signed_int CLOSE_PAREN
             '''
diff --git a/astropy/units/format/generic_parsetab.py b/astropy/units/format/generic_parsetab.py
--- a/astropy/units/format/generic_parsetab.py
+++ b/astropy/units/format/generic_parsetab.py
@@ -16,9 +16,9 @@
 
 _lr_method = 'LALR'
 
-_lr_signature = 'DOUBLE_STAR STAR PERIOD SOLIDUS CARET OPEN_PAREN CLOSE_PAREN FUNCNAME UNIT SIGN UINT UFLOAT\n            main : product_of_units\n                 | factor product_of_units\n                 | factor product product_of_units\n                 | division_product_of_units\n                 | factor division_product_of_units\n                 | factor product division_product_of_units\n                 | inverse_unit\n                 | factor inverse_unit\n                 | factor product inverse_unit\n                 | factor\n            \n            division_product_of_units : division_product_of_units division product_of_units\n                                      | product_of_units\n            \n            inverse_unit : division unit_expression\n            \n            factor : factor_fits\n                   | factor_float\n                   | factor_int\n            \n            factor_float : signed_float\n                         | signed_float UINT signed_int\n                         | signed_float UINT power numeric_power\n            \n            factor_int : UINT\n                       | UINT signed_int\n                       | UINT power numeric_power\n                       | UINT UINT signed_int\n                       | UINT UINT power numeric_power\n            \n            factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN\n                        | UINT power signed_int\n                        | UINT SIGN UINT\n                        | UINT OPEN_PAREN signed_int CLOSE_PAREN\n            \n            product_of_units : unit_expression product product_of_units\n                             | unit_expression product_of_units\n                             | unit_expression\n            \n            unit_expression : function\n                            | unit_with_power\n                            | OPEN_PAREN product_of_units CLOSE_PAREN\n            \n            unit_with_power : UNIT power numeric_power\n                            | UNIT numeric_power\n                            | UNIT\n            \n            numeric_power : sign UINT\n                          | OPEN_PAREN paren_expr CLOSE_PAREN\n            \n            paren_expr : sign UINT\n                       | signed_float\n                       | frac\n            \n            frac : sign UINT division sign UINT\n            \n            sign : SIGN\n                 |\n            \n            product : STAR\n                    | PERIOD\n            \n            division : SOLIDUS\n            \n            power : DOUBLE_STAR\n                  | CARET\n            \n            signed_int : SIGN UINT\n            \n            signed_float : sign UINT\n                         | sign UFLOAT\n            \n            function_name : FUNCNAME\n            \n            function : function_name OPEN_PAREN main CLOSE_PAREN\n            '
+_lr_signature = 'DOUBLE_STAR STAR PERIOD SOLIDUS CARET OPEN_PAREN CLOSE_PAREN FUNCNAME UNIT SIGN UINT UFLOAT\n            main : product_of_units\n                 | factor product_of_units\n                 | factor product product_of_units\n                 | division_product_of_units\n                 | factor division_product_of_units\n                 | factor product division_product_of_units\n                 | inverse_unit\n                 | factor inverse_unit\n                 | factor product inverse_unit\n                 | factor\n            \n            division_product_of_units : division_product_of_units division product_of_units\n                                      | product_of_units\n            \n            inverse_unit : division unit_expression\n            \n            factor : factor_fits\n                   | factor_float\n                   | factor_int\n            \n            factor_float : signed_float\n                         | signed_float UINT signed_int\n                         | signed_float UINT power numeric_power\n            \n            factor_int : UINT\n                       | UINT signed_int\n                       | UINT power numeric_power\n                       | UINT UINT signed_int\n                       | UINT UINT power numeric_power\n            \n            factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN\n                        | UINT power OPEN_PAREN UINT CLOSE_PAREN\n                        | UINT power signed_int\n                        | UINT power UINT\n                        | UINT SIGN UINT\n                        | UINT OPEN_PAREN signed_int CLOSE_PAREN\n            \n            product_of_units : unit_expression product product_of_units\n                             | unit_expression product_of_units\n                             | unit_expression\n            \n            unit_expression : function\n                            | unit_with_power\n                            | OPEN_PAREN product_of_units CLOSE_PAREN\n            \n            unit_with_power : UNIT power numeric_power\n                            | UNIT numeric_power\n                            | UNIT\n            \n            numeric_power : sign UINT\n                          | OPEN_PAREN paren_expr CLOSE_PAREN\n            \n            paren_expr : sign UINT\n                       | signed_float\n                       | frac\n            \n            frac : sign UINT division sign UINT\n            \n            sign : SIGN\n                 |\n            \n            product : STAR\n                    | PERIOD\n            \n            division : SOLIDUS\n            \n            power : DOUBLE_STAR\n                  | CARET\n            \n            signed_int : SIGN UINT\n            \n            signed_float : sign UINT\n                         | sign UFLOAT\n            \n            function_name : FUNCNAME\n            \n            function : function_name OPEN_PAREN main CLOSE_PAREN\n            '
     
-_lr_action_items = {'OPEN_PAREN':([0,3,6,7,8,9,10,11,12,13,14,16,17,18,19,21,23,26,27,28,29,34,36,38,39,41,42,43,46,47,53,54,55,58,59,62,63,64,66,67,72,73,75,76,77,78,80,],[13,13,13,-14,-15,-16,13,-32,-33,13,35,-17,-48,41,45,-54,13,-46,-47,13,13,57,-21,-49,-50,13,45,-36,-52,-53,-34,-23,45,-26,-22,-27,-18,45,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'UINT':([0,14,15,16,17,19,20,34,37,38,39,41,42,44,45,46,47,55,56,57,60,64,69,81,82,],[14,33,-44,40,-48,-45,46,-45,62,-49,-50,14,-45,67,-45,-52,-53,-45,73,-45,73,-45,79,-45,83,]),'SOLIDUS':([0,2,3,4,6,7,8,9,11,12,14,16,19,22,23,24,26,27,30,36,41,43,46,47,48,49,51,52,53,54,58,59,62,63,66,67,72,73,75,76,77,78,79,80,],[17,-12,17,17,-31,-14,-15,-16,-32,-33,-20,-17,-37,-12,17,17,-46,-47,-30,-21,17,-36,-52,-53,-12,17,-11,-29,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,17,-25,]),'UNIT':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[19,19,19,-14,-15,-16,19,-32,-33,19,-20,-17,-48,-37,19,-46,-47,19,19,-21,19,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'FUNCNAME':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[21,21,21,-14,-15,-16,21,-32,-33,21,-20,-17,-48,-37,21,-46,-47,21,21,-21,21,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'SIGN':([0,14,17,19,33,34,35,38,39,40,41,42,45,55,57,64,81,],[15,37,-48,15,56,60,56,-49,-50,56,15,15,15,15,60,15,15,]),'UFLOAT':([0,15,20,41,45,57,60,69,],[-45,-44,47,-45,-45,-45,-44,47,]),'$end':([1,2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,36,43,46,47,48,49,50,51,52,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[0,-1,-10,-4,-7,-31,-14,-15,-16,-32,-33,-20,-17,-37,-2,-5,-8,-30,-13,-21,-36,-52,-53,-3,-6,-9,-11,-29,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'CLOSE_PAREN':([2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,32,36,43,46,47,48,49,50,51,52,53,54,58,59,61,62,63,65,66,67,68,70,71,72,73,74,75,76,77,78,79,80,83,],[-1,-10,-4,-7,-31,-14,-15,-16,-32,-33,-20,-17,-37,-2,-5,-8,-30,-13,53,-21,-36,-52,-53,-3,-6,-9,-11,-29,-34,-23,-26,-22,75,-27,-18,77,-35,-38,78,-41,-42,-24,-51,80,-28,-19,-55,-39,-40,-25,-43,]),'STAR':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[26,26,-14,-15,-16,-32,-33,-20,-17,-37,-21,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'PERIOD':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[27,27,-14,-15,-16,-32,-33,-20,-17,-37,-21,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'DOUBLE_STAR':([14,19,33,40,],[38,38,38,38,]),'CARET':([14,19,33,40,],[39,39,39,39,]),}
+_lr_action_items = {'OPEN_PAREN':([0,3,6,7,8,9,10,11,12,13,14,16,17,18,19,21,23,26,27,28,29,34,36,38,39,41,42,43,46,47,53,54,55,57,59,60,63,64,65,67,68,73,74,77,78,79,80,82,83,],[13,13,13,-14,-15,-16,13,-34,-35,13,35,-17,-50,41,45,-56,13,-48,-49,13,13,58,-21,-51,-52,13,45,-38,-54,-55,-36,-23,45,-28,-27,-22,-29,-18,45,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'UINT':([0,14,15,16,17,19,20,34,37,38,39,41,42,44,45,46,47,55,56,58,61,65,70,84,85,],[14,33,-46,40,-50,-47,46,57,63,-51,-52,14,-47,68,-47,-54,-55,-47,74,75,74,-47,81,-47,86,]),'SOLIDUS':([0,2,3,4,6,7,8,9,11,12,14,16,19,22,23,24,26,27,30,36,41,43,46,47,48,49,51,52,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,81,82,83,],[17,-12,17,17,-33,-14,-15,-16,-34,-35,-20,-17,-39,-12,17,17,-48,-49,-32,-21,17,-38,-54,-55,-12,17,-11,-31,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,17,-26,-25,]),'UNIT':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[19,19,19,-14,-15,-16,19,-34,-35,19,-20,-17,-50,-39,19,-48,-49,19,19,-21,19,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'FUNCNAME':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[21,21,21,-14,-15,-16,21,-34,-35,21,-20,-17,-50,-39,21,-48,-49,21,21,-21,21,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'SIGN':([0,14,17,19,33,34,35,38,39,40,41,42,45,55,58,65,84,],[15,37,-50,15,56,61,56,-51,-52,56,15,15,15,15,61,15,15,]),'UFLOAT':([0,15,20,41,45,58,61,70,],[-47,-46,47,-47,-47,-47,-46,47,]),'$end':([1,2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,36,43,46,47,48,49,50,51,52,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[0,-1,-10,-4,-7,-33,-14,-15,-16,-34,-35,-20,-17,-39,-2,-5,-8,-32,-13,-21,-38,-54,-55,-3,-6,-9,-11,-31,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'CLOSE_PAREN':([2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,32,36,43,46,47,48,49,50,51,52,53,54,57,59,60,62,63,64,66,67,68,69,71,72,73,74,75,76,77,78,79,80,81,82,83,86,],[-1,-10,-4,-7,-33,-14,-15,-16,-34,-35,-20,-17,-39,-2,-5,-8,-32,-13,53,-21,-38,-54,-55,-3,-6,-9,-11,-31,-36,-23,-28,-27,-22,77,-29,-18,79,-37,-40,80,-43,-44,-24,-53,82,83,-30,-19,-57,-41,-42,-26,-25,-45,]),'STAR':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[26,26,-14,-15,-16,-34,-35,-20,-17,-39,-21,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'PERIOD':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[27,27,-14,-15,-16,-34,-35,-20,-17,-39,-21,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'DOUBLE_STAR':([14,19,33,40,],[38,38,38,38,]),'CARET':([14,19,33,40,],[39,39,39,39,]),}
 
 _lr_action = {}
 for _k, _v in _lr_action_items.items():
@@ -27,7 +27,7 @@
       _lr_action[_x][_k] = _y
 del _lr_action_items
 
-_lr_goto_items = {'main':([0,41,],[1,65,]),'product_of_units':([0,3,6,13,23,28,29,41,],[2,22,30,32,48,51,52,2,]),'factor':([0,41,],[3,3,]),'division_product_of_units':([0,3,23,41,],[4,24,49,4,]),'inverse_unit':([0,3,23,41,],[5,25,50,5,]),'unit_expression':([0,3,6,10,13,23,28,29,41,],[6,6,6,31,6,6,6,6,6,]),'factor_fits':([0,41,],[7,7,]),'factor_float':([0,41,],[8,8,]),'factor_int':([0,41,],[9,9,]),'division':([0,3,4,23,24,41,49,79,],[10,10,28,10,28,10,28,81,]),'function':([0,3,6,10,13,23,28,29,41,],[11,11,11,11,11,11,11,11,11,]),'unit_with_power':([0,3,6,10,13,23,28,29,41,],[12,12,12,12,12,12,12,12,12,]),'signed_float':([0,41,45,57,],[16,16,70,70,]),'function_name':([0,3,6,10,13,23,28,29,41,],[18,18,18,18,18,18,18,18,18,]),'sign':([0,19,34,41,42,45,55,57,64,81,],[20,44,44,20,44,69,44,69,44,82,]),'product':([3,6,],[23,29,]),'power':([14,19,33,40,],[34,42,55,64,]),'signed_int':([14,33,34,35,40,57,],[36,54,58,61,63,74,]),'numeric_power':([19,34,42,55,64,],[43,59,66,72,76,]),'paren_expr':([45,57,],[68,68,]),'frac':([45,57,],[71,71,]),}
+_lr_goto_items = {'main':([0,41,],[1,66,]),'product_of_units':([0,3,6,13,23,28,29,41,],[2,22,30,32,48,51,52,2,]),'factor':([0,41,],[3,3,]),'division_product_of_units':([0,3,23,41,],[4,24,49,4,]),'inverse_unit':([0,3,23,41,],[5,25,50,5,]),'unit_expression':([0,3,6,10,13,23,28,29,41,],[6,6,6,31,6,6,6,6,6,]),'factor_fits':([0,41,],[7,7,]),'factor_float':([0,41,],[8,8,]),'factor_int':([0,41,],[9,9,]),'division':([0,3,4,23,24,41,49,81,],[10,10,28,10,28,10,28,84,]),'function':([0,3,6,10,13,23,28,29,41,],[11,11,11,11,11,11,11,11,11,]),'unit_with_power':([0,3,6,10,13,23,28,29,41,],[12,12,12,12,12,12,12,12,12,]),'signed_float':([0,41,45,58,],[16,16,71,71,]),'function_name':([0,3,6,10,13,23,28,29,41,],[18,18,18,18,18,18,18,18,18,]),'sign':([0,19,34,41,42,45,55,58,65,84,],[20,44,44,20,44,70,44,70,44,85,]),'product':([3,6,],[23,29,]),'power':([14,19,33,40,],[34,42,55,65,]),'signed_int':([14,33,34,35,40,58,],[36,54,59,62,64,76,]),'numeric_power':([19,34,42,55,65,],[43,60,67,73,78,]),'paren_expr':([45,58,],[69,69,]),'frac':([45,58,],[72,72,]),}
 
 _lr_goto = {}
 for _k, _v in _lr_goto_items.items():
@@ -62,34 +62,36 @@
   ('factor_int -> UINT UINT signed_int','factor_int',3,'p_factor_int','generic.py',257),
   ('factor_int -> UINT UINT power numeric_power','factor_int',4,'p_factor_int','generic.py',258),
   ('factor_fits -> UINT power OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',5,'p_factor_fits','generic.py',276),
-  ('factor_fits -> UINT power signed_int','factor_fits',3,'p_factor_fits','generic.py',277),
-  ('factor_fits -> UINT SIGN UINT','factor_fits',3,'p_factor_fits','generic.py',278),
-  ('factor_fits -> UINT OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',4,'p_factor_fits','generic.py',279),
-  ('product_of_units -> unit_expression product product_of_units','product_of_units',3,'p_product_of_units','generic.py',298),
-  ('product_of_units -> unit_expression product_of_units','product_of_units',2,'p_product_of_units','generic.py',299),
-  ('product_of_units -> unit_expression','product_of_units',1,'p_product_of_units','generic.py',300),
-  ('unit_expression -> function','unit_expression',1,'p_unit_expression','generic.py',311),
-  ('unit_expression -> unit_with_power','unit_expression',1,'p_unit_expression','generic.py',312),
-  ('unit_expression -> OPEN_PAREN product_of_units CLOSE_PAREN','unit_expression',3,'p_unit_expression','generic.py',313),
-  ('unit_with_power -> UNIT power numeric_power','unit_with_power',3,'p_unit_with_power','generic.py',322),
-  ('unit_with_power -> UNIT numeric_power','unit_with_power',2,'p_unit_with_power','generic.py',323),
-  ('unit_with_power -> UNIT','unit_with_power',1,'p_unit_with_power','generic.py',324),
-  ('numeric_power -> sign UINT','numeric_power',2,'p_numeric_power','generic.py',335),
-  ('numeric_power -> OPEN_PAREN paren_expr CLOSE_PAREN','numeric_power',3,'p_numeric_power','generic.py',336),
-  ('paren_expr -> sign UINT','paren_expr',2,'p_paren_expr','generic.py',345),
-  ('paren_expr -> signed_float','paren_expr',1,'p_paren_expr','generic.py',346),
-  ('paren_expr -> frac','paren_expr',1,'p_paren_expr','generic.py',347),
-  ('frac -> sign UINT division sign UINT','frac',5,'p_frac','generic.py',356),
-  ('sign -> SIGN','sign',1,'p_sign','generic.py',362),
-  ('sign -> <empty>','sign',0,'p_sign','generic.py',363),
-  ('product -> STAR','product',1,'p_product','generic.py',372),
-  ('product -> PERIOD','product',1,'p_product','generic.py',373),
-  ('division -> SOLIDUS','division',1,'p_division','generic.py',379),
-  ('power -> DOUBLE_STAR','power',1,'p_power','generic.py',385),
-  ('power -> CARET','power',1,'p_power','generic.py',386),
-  ('signed_int -> SIGN UINT','signed_int',2,'p_signed_int','generic.py',392),
-  ('signed_float -> sign UINT','signed_float',2,'p_signed_float','generic.py',398),
-  ('signed_float -> sign UFLOAT','signed_float',2,'p_signed_float','generic.py',399),
-  ('function_name -> FUNCNAME','function_name',1,'p_function_name','generic.py',405),
-  ('function -> function_name OPEN_PAREN main CLOSE_PAREN','function',4,'p_function','generic.py',411),
+  ('factor_fits -> UINT power OPEN_PAREN UINT CLOSE_PAREN','factor_fits',5,'p_factor_fits','generic.py',277),
+  ('factor_fits -> UINT power signed_int','factor_fits',3,'p_factor_fits','generic.py',278),
+  ('factor_fits -> UINT power UINT','factor_fits',3,'p_factor_fits','generic.py',279),
+  ('factor_fits -> UINT SIGN UINT','factor_fits',3,'p_factor_fits','generic.py',280),
+  ('factor_fits -> UINT OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',4,'p_factor_fits','generic.py',281),
+  ('product_of_units -> unit_expression product product_of_units','product_of_units',3,'p_product_of_units','generic.py',300),
+  ('product_of_units -> unit_expression product_of_units','product_of_units',2,'p_product_of_units','generic.py',301),
+  ('product_of_units -> unit_expression','product_of_units',1,'p_product_of_units','generic.py',302),
+  ('unit_expression -> function','unit_expression',1,'p_unit_expression','generic.py',313),
+  ('unit_expression -> unit_with_power','unit_expression',1,'p_unit_expression','generic.py',314),
+  ('unit_expression -> OPEN_PAREN product_of_units CLOSE_PAREN','unit_expression',3,'p_unit_expression','generic.py',315),
+  ('unit_with_power -> UNIT power numeric_power','unit_with_power',3,'p_unit_with_power','generic.py',324),
+  ('unit_with_power -> UNIT numeric_power','unit_with_power',2,'p_unit_with_power','generic.py',325),
+  ('unit_with_power -> UNIT','unit_with_power',1,'p_unit_with_power','generic.py',326),
+  ('numeric_power -> sign UINT','numeric_power',2,'p_numeric_power','generic.py',337),
+  ('numeric_power -> OPEN_PAREN paren_expr CLOSE_PAREN','numeric_power',3,'p_numeric_power','generic.py',338),
+  ('paren_expr -> sign UINT','paren_expr',2,'p_paren_expr','generic.py',347),
+  ('paren_expr -> signed_float','paren_expr',1,'p_paren_expr','generic.py',348),
+  ('paren_expr -> frac','paren_expr',1,'p_paren_expr','generic.py',349),
+  ('frac -> sign UINT division sign UINT','frac',5,'p_frac','generic.py',358),
+  ('sign -> SIGN','sign',1,'p_sign','generic.py',364),
+  ('sign -> <empty>','sign',0,'p_sign','generic.py',365),
+  ('product -> STAR','product',1,'p_product','generic.py',374),
+  ('product -> PERIOD','product',1,'p_product','generic.py',375),
+  ('division -> SOLIDUS','division',1,'p_division','generic.py',381),
+  ('power -> DOUBLE_STAR','power',1,'p_power','generic.py',387),
+  ('power -> CARET','power',1,'p_power','generic.py',388),
+  ('signed_int -> SIGN UINT','signed_int',2,'p_signed_int','generic.py',394),
+  ('signed_float -> sign UINT','signed_float',2,'p_signed_float','generic.py',400),
+  ('signed_float -> sign UFLOAT','signed_float',2,'p_signed_float','generic.py',401),
+  ('function_name -> FUNCNAME','function_name',1,'p_function_name','generic.py',407),
+  ('function -> function_name OPEN_PAREN main CLOSE_PAREN','function',4,'p_function','generic.py',413),
 ]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/units/format/generic.py | 277 | 277 | 7 | 4 | 2692
| astropy/units/format/generic_parsetab.py | 19 | 21 | 36 | 5 | 17076
| astropy/units/format/generic_parsetab.py | 30 | 30 | - | 5 | -
| astropy/units/format/generic_parsetab.py | 65 | 94 | 9 | 5 | 4376


## Problem Statement

```
FITS-standard unit parsing fails on some types of exponents
Why don't these work:
\`\`\`python
from astropy.units import Unit
Unit('10**17 erg/(cm2 s Angstrom)', format='fits')
Unit('10^17 erg/(cm2 s Angstrom)', format='fits')
\`\`\`
When these all do:
\`\`\`python
from astropy.units import Unit
Unit('10+17 erg/(cm2 s Angstrom)', format='fits')
Unit('10**-17 erg/(cm2 s Angstrom)', format='fits')
Unit('10^-17 erg/(cm2 s Angstrom)', format='fits')
Unit('10-17 erg/(cm2 s Angstrom)', format='fits')
\`\`\`

The non-working versions give *e.g.*:
\`\`\`
ValueError: '10^17 erg/(cm2 s Angstrom)' did not parse as fits unit: Numeric factor not supported by FITS
\`\`\`
which is not how I would interpret the [FITS standard](https://fits.gsfc.nasa.gov/standard30/fits_standard30aa.pdf).

Tested on 2.0.7 and 3.0.3

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 astropy/units/format/fits.py | 82 | 108| 199 | 199 | 1269 | 
| 2 | 1 astropy/units/format/fits.py | 139 | 158| 145 | 344 | 1269 | 
| 3 | 1 astropy/units/format/fits.py | 110 | 137| 228 | 572 | 1269 | 
| 4 | 1 astropy/units/format/fits.py | 8 | 80| 684 | 1256 | 1269 | 
| 5 | 2 astropy/units/astrophys.py | 78 | 154| 752 | 2008 | 3110 | 
| 6 | 3 astropy/units/core.py | 1693 | 1755| 471 | 2479 | 20270 | 
| **-> 7 <-** | **4 astropy/units/format/generic.py** | 274 | 294| 213 | 2692 | 23873 | 
| 8 | **4 astropy/units/format/generic.py** | 440 | 495| 381 | 3073 | 23873 | 
| **-> 9 <-** | **5 astropy/units/format/generic_parsetab.py** | 38 | 96| 1303 | 4376 | 28119 | 
| 10 | 6 astropy/units/quantity_helper/erfa.py | 1 | 43| 333 | 4709 | 28703 | 
| 11 | 7 astropy/units/format/cds_parsetab.py | 38 | 65| 589 | 5298 | 30510 | 
| 12 | **7 astropy/units/format/generic.py** | 252 | 272| 222 | 5520 | 30510 | 
| 13 | 8 astropy/io/fits/column.py | 78 | 126| 756 | 6276 | 52413 | 
| 14 | 9 astropy/units/physical.py | 69 | 134| 837 | 7113 | 53618 | 
| 15 | 10 astropy/units/si.py | 1 | 82| 734 | 7847 | 55742 | 
| 16 | **10 astropy/units/format/generic.py** | 55 | 103| 270 | 8117 | 55742 | 
| 17 | 11 astropy/units/quantity_helper/helpers.py | 87 | 195| 762 | 8879 | 59376 | 
| 18 | 11 astropy/units/astrophys.py | 155 | 189| 313 | 9192 | 59376 | 
| 19 | **11 astropy/units/format/generic.py** | 237 | 250| 160 | 9352 | 59376 | 
| 20 | 12 astropy/units/imperial.py | 1 | 107| 754 | 10106 | 60667 | 
| 21 | 12 astropy/units/si.py | 84 | 177| 757 | 10863 | 60667 | 
| 22 | 13 astropy/units/format/ogip_parsetab.py | 38 | 82| 1026 | 11889 | 63824 | 
| 23 | 14 astropy/units/format/vounit.py | 82 | 96| 131 | 12020 | 65707 | 
| 24 | 14 astropy/units/core.py | 2156 | 2187| 401 | 12421 | 65707 | 
| 25 | 14 astropy/units/quantity_helper/helpers.py | 273 | 338| 757 | 13178 | 65707 | 
| 26 | 14 astropy/units/quantity_helper/helpers.py | 198 | 257| 483 | 13661 | 65707 | 
| 27 | 14 astropy/units/core.py | 892 | 920| 244 | 13905 | 65707 | 
| 28 | 14 astropy/units/format/vounit.py | 98 | 122| 185 | 14090 | 65707 | 
| 29 | 14 astropy/units/format/vounit.py | 186 | 236| 366 | 14456 | 65707 | 
| 30 | 15 astropy/units/format/ogip.py | 1 | 50| 165 | 14621 | 69345 | 
| 31 | 15 astropy/units/format/ogip.py | 402 | 423| 165 | 14786 | 69345 | 
| 32 | 15 astropy/units/core.py | 761 | 779| 143 | 14929 | 69345 | 
| 33 | 16 astropy/units/quantity_helper/scipy_special.py | 1 | 59| 691 | 15620 | 70374 | 
| 34 | 16 astropy/units/format/ogip.py | 375 | 400| 200 | 15820 | 70374 | 
| 35 | 16 astropy/units/astrophys.py | 1 | 77| 733 | 16553 | 70374 | 
| **-> 36 <-** | **16 astropy/units/format/generic_parsetab.py** | 1 | 21| 523 | 17076 | 70374 | 
| 37 | 16 astropy/units/format/ogip_parsetab.py | 1 | 19| 413 | 17489 | 70374 | 
| 38 | 16 astropy/units/si.py | 178 | 243| 591 | 18080 | 70374 | 
| 39 | 17 astropy/units/cgs.py | 1 | 136| 850 | 18930 | 71268 | 
| 40 | 18 astropy/units/format/utils.py | 113 | 149| 226 | 19156 | 72651 | 
| 41 | 19 astropy/io/votable/exceptions.py | 1023 | 1034| 104 | 19260 | 85120 | 
| 42 | 20 astropy/units/format/cds.py | 205 | 224| 196 | 19456 | 87550 | 
| 43 | 21 astropy/units/function/units.py | 1 | 47| 281 | 19737 | 87927 | 
| 44 | 21 astropy/units/imperial.py | 108 | 168| 386 | 20123 | 87927 | 
| 45 | 21 astropy/units/format/ogip.py | 52 | 110| 589 | 20712 | 87927 | 
| 46 | 21 astropy/units/core.py | 1 | 32| 199 | 20911 | 87927 | 
| 47 | 22 astropy/modeling/functional_models.py | 2361 | 2378| 189 | 21100 | 107268 | 
| 48 | 23 astropy/units/equivalencies.py | 61 | 76| 146 | 21246 | 115422 | 
| 49 | 23 astropy/units/format/cds_parsetab.py | 1 | 23| 748 | 21994 | 115422 | 
| 50 | 23 astropy/units/format/utils.py | 45 | 75| 217 | 22211 | 115422 | 
| 51 | 23 astropy/units/format/cds.py | 280 | 301| 146 | 22357 | 115422 | 
| 52 | 24 astropy/time/formats.py | 1016 | 1053| 580 | 22937 | 127662 | 
| 53 | 25 astropy/units/quantity_helper/converters.py | 104 | 122| 127 | 23064 | 130778 | 
| 54 | 25 astropy/units/format/ogip.py | 465 | 480| 160 | 23224 | 130778 | 
| 55 | 26 astropy/units/function/core.py | 259 | 283| 209 | 23433 | 136301 | 
| 56 | 26 astropy/units/format/ogip.py | 230 | 266| 401 | 23834 | 136301 | 
| 57 | 26 astropy/units/format/vounit.py | 124 | 152| 249 | 24083 | 136301 | 
| 58 | 26 astropy/units/format/vounit.py | 28 | 80| 594 | 24677 | 136301 | 
| 59 | 26 astropy/units/format/ogip.py | 449 | 463| 145 | 24822 | 136301 | 
| 60 | 27 astropy/units/format/__init__.py | 11 | 27| 135 | 24957 | 136744 | 
| 61 | 27 astropy/units/quantity_helper/helpers.py | 45 | 84| 346 | 25303 | 136744 | 
| 62 | 27 astropy/units/function/core.py | 303 | 332| 235 | 25538 | 136744 | 
| 63 | 27 astropy/units/format/cds.py | 303 | 337| 240 | 25778 | 136744 | 
| 64 | 27 astropy/units/quantity_helper/helpers.py | 260 | 270| 113 | 25891 | 136744 | 
| 65 | 28 astropy/io/misc/asdf/tags/unit/unit.py | 4 | 26| 169 | 26060 | 136938 | 
| 66 | 28 astropy/units/core.py | 452 | 485| 171 | 26231 | 136938 | 
| 67 | 28 astropy/units/format/ogip.py | 425 | 447| 176 | 26407 | 136938 | 
| 68 | 28 astropy/units/core.py | 1243 | 1269| 276 | 26683 | 136938 | 
| 69 | 28 astropy/units/core.py | 742 | 759| 133 | 26816 | 136938 | 
| 70 | 28 astropy/units/core.py | 616 | 641| 184 | 27000 | 136938 | 
| 71 | **28 astropy/units/format/generic.py** | 191 | 210| 183 | 27183 | 136938 | 
| 72 | 29 astropy/units/format/generic_lextab.py | 1 | 22| 443 | 27626 | 137489 | 
| 73 | 29 astropy/units/format/ogip_parsetab.py | 21 | 21| 1114 | 28740 | 137489 | 
| 74 | 29 astropy/units/function/core.py | 334 | 360| 272 | 29012 | 137489 | 
| 75 | 30 astropy/units/function/__init__.py | 1 | 11| 79 | 29091 | 137568 | 
| 76 | 30 astropy/units/format/ogip.py | 268 | 285| 197 | 29288 | 137568 | 
| 77 | 31 astropy/io/fits/util.py | 276 | 309| 355 | 29643 | 144520 | 
| 78 | 32 astropy/coordinates/errors.py | 154 | 176| 159 | 29802 | 145610 | 
| 79 | 32 astropy/units/format/cds.py | 339 | 372| 242 | 30044 | 145610 | 
| 80 | 33 astropy/units/format/ogip_lextab.py | 1 | 22| 453 | 30497 | 146172 | 
| 81 | 34 astropy/modeling/core.py | 1491 | 1565| 662 | 31159 | 174151 | 
| 82 | 34 astropy/units/format/cds.py | 1 | 78| 372 | 31531 | 174151 | 
| 83 | 35 astropy/units/format/cds_lextab.py | 1 | 22| 318 | 31849 | 174577 | 
| 84 | 35 astropy/time/formats.py | 1 | 41| 404 | 32253 | 174577 | 
| 85 | 35 astropy/io/fits/column.py | 3 | 77| 761 | 33014 | 174577 | 
| 86 | 35 astropy/units/quantity_helper/converters.py | 159 | 270| 1143 | 34157 | 174577 | 
| 87 | 35 astropy/units/format/vounit.py | 154 | 184| 230 | 34387 | 174577 | 
| 88 | 35 astropy/units/core.py | 1201 | 1216| 183 | 34570 | 174577 | 
| 89 | 35 astropy/units/function/core.py | 285 | 301| 155 | 34725 | 174577 | 
| 90 | **35 astropy/units/format/generic.py** | 497 | 528| 215 | 34940 | 174577 | 
| 91 | 35 astropy/units/function/core.py | 1 | 25| 220 | 35160 | 174577 | 
| 92 | 35 astropy/units/core.py | 1218 | 1241| 229 | 35389 | 174577 | 
| 93 | 35 astropy/units/quantity_helper/erfa.py | 46 | 63| 224 | 35613 | 174577 | 
| 94 | 35 astropy/modeling/functional_models.py | 2483 | 2502| 198 | 35811 | 174577 | 
| 95 | 36 astropy/units/format/latex.py | 87 | 123| 217 | 36028 | 175462 | 
| 96 | 36 astropy/units/format/utils.py | 191 | 219| 214 | 36242 | 175462 | 
| 97 | 36 astropy/units/format/cds.py | 226 | 278| 330 | 36572 | 175462 | 
| 98 | 37 astropy/units/quantity.py | 1 | 55| 392 | 36964 | 189634 | 
| 99 | 37 astropy/units/equivalencies.py | 4 | 26| 188 | 37152 | 189634 | 
| 100 | 37 astropy/units/core.py | 64 | 103| 285 | 37437 | 189634 | 
| 101 | **37 astropy/units/format/generic.py** | 296 | 407| 692 | 38129 | 189634 | 
| 102 | 37 astropy/units/core.py | 843 | 890| 369 | 38498 | 189634 | 
| 103 | 37 astropy/units/core.py | 703 | 719| 140 | 38638 | 189634 | 
| 104 | 37 astropy/units/format/ogip.py | 322 | 335| 144 | 38782 | 189634 | 
| 105 | 37 astropy/modeling/functional_models.py | 430 | 449| 210 | 38992 | 189634 | 
| 106 | **37 astropy/units/format/generic.py** | 173 | 189| 162 | 39154 | 189634 | 
| 107 | **37 astropy/units/format/generic.py** | 212 | 235| 178 | 39332 | 189634 | 
| 108 | 38 astropy/units/format/unicode_format.py | 47 | 70| 178 | 39510 | 190032 | 
| 109 | 38 astropy/units/equivalencies.py | 676 | 685| 172 | 39682 | 190032 | 


### Hint

```
Additional examples that *do* work:
\`\`\`python
Unit('10**+17 erg/(cm2 s Angstrom)', format='fits')
Unit('10^+17 erg/(cm2 s Angstrom)', format='fits')
\`\`\`
It seems that currently the sign is always required for the `**` and `^`, though it should not:

> The final units string is the compound string, or a compound of compounds, preceded by an optional numeric multiplier of the form 10**k, 10ˆk, or 10±k where k is an integer, optionally surrounded by parentheses with the sign character required in the third form in the absence of parentheses.

> The power may be a simple integer, with or without sign, optionally surrounded by parentheses.
The place to look in the parser is https://github.com/astropy/astropy/blob/master/astropy/units/format/generic.py#L274, and I think all it would take is replace `signed_int` by `numeric_power` (but don't have time to try myself right now).
I tried two possibilities:

1. Simply replace `UINT power signed_int` with `UINT power numeric_power`.  That broke valid expressions like `10**+2`.
2. Add `UINT power numeric_power` in addition to `UINT power signed_int`.  That did not make `10**2` valid.
I think it may have to be `UINT power SIGN numeric_power` - sign can be empty.
Unfortunately that didn't help either, it broke the existing valid expressions and did not make `10**2` valid.
Another odd thing. In the traceback of the test failures I can see [p_factor_int()](https://github.com/astropy/astropy/blob/master/astropy/units/format/generic.py#L252) being called but not [p_factor_fits()](https://github.com/astropy/astropy/blob/master/astropy/units/format/generic.py#L274).
@weaverba137 - that last thing at least is probably not odd: the test fails because in its current form `p_factor_fits()` does not match the string.

On why my suggestions do not work: I'm a bit at a loss and will try to investigate, though I'm not quite sure when...
```

## Patch

```diff
diff --git a/astropy/units/format/generic.py b/astropy/units/format/generic.py
--- a/astropy/units/format/generic.py
+++ b/astropy/units/format/generic.py
@@ -274,7 +274,9 @@ def p_factor_int(p):
         def p_factor_fits(p):
             '''
             factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN
+                        | UINT power OPEN_PAREN UINT CLOSE_PAREN
                         | UINT power signed_int
+                        | UINT power UINT
                         | UINT SIGN UINT
                         | UINT OPEN_PAREN signed_int CLOSE_PAREN
             '''
diff --git a/astropy/units/format/generic_parsetab.py b/astropy/units/format/generic_parsetab.py
--- a/astropy/units/format/generic_parsetab.py
+++ b/astropy/units/format/generic_parsetab.py
@@ -16,9 +16,9 @@
 
 _lr_method = 'LALR'
 
-_lr_signature = 'DOUBLE_STAR STAR PERIOD SOLIDUS CARET OPEN_PAREN CLOSE_PAREN FUNCNAME UNIT SIGN UINT UFLOAT\n            main : product_of_units\n                 | factor product_of_units\n                 | factor product product_of_units\n                 | division_product_of_units\n                 | factor division_product_of_units\n                 | factor product division_product_of_units\n                 | inverse_unit\n                 | factor inverse_unit\n                 | factor product inverse_unit\n                 | factor\n            \n            division_product_of_units : division_product_of_units division product_of_units\n                                      | product_of_units\n            \n            inverse_unit : division unit_expression\n            \n            factor : factor_fits\n                   | factor_float\n                   | factor_int\n            \n            factor_float : signed_float\n                         | signed_float UINT signed_int\n                         | signed_float UINT power numeric_power\n            \n            factor_int : UINT\n                       | UINT signed_int\n                       | UINT power numeric_power\n                       | UINT UINT signed_int\n                       | UINT UINT power numeric_power\n            \n            factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN\n                        | UINT power signed_int\n                        | UINT SIGN UINT\n                        | UINT OPEN_PAREN signed_int CLOSE_PAREN\n            \n            product_of_units : unit_expression product product_of_units\n                             | unit_expression product_of_units\n                             | unit_expression\n            \n            unit_expression : function\n                            | unit_with_power\n                            | OPEN_PAREN product_of_units CLOSE_PAREN\n            \n            unit_with_power : UNIT power numeric_power\n                            | UNIT numeric_power\n                            | UNIT\n            \n            numeric_power : sign UINT\n                          | OPEN_PAREN paren_expr CLOSE_PAREN\n            \n            paren_expr : sign UINT\n                       | signed_float\n                       | frac\n            \n            frac : sign UINT division sign UINT\n            \n            sign : SIGN\n                 |\n            \n            product : STAR\n                    | PERIOD\n            \n            division : SOLIDUS\n            \n            power : DOUBLE_STAR\n                  | CARET\n            \n            signed_int : SIGN UINT\n            \n            signed_float : sign UINT\n                         | sign UFLOAT\n            \n            function_name : FUNCNAME\n            \n            function : function_name OPEN_PAREN main CLOSE_PAREN\n            '
+_lr_signature = 'DOUBLE_STAR STAR PERIOD SOLIDUS CARET OPEN_PAREN CLOSE_PAREN FUNCNAME UNIT SIGN UINT UFLOAT\n            main : product_of_units\n                 | factor product_of_units\n                 | factor product product_of_units\n                 | division_product_of_units\n                 | factor division_product_of_units\n                 | factor product division_product_of_units\n                 | inverse_unit\n                 | factor inverse_unit\n                 | factor product inverse_unit\n                 | factor\n            \n            division_product_of_units : division_product_of_units division product_of_units\n                                      | product_of_units\n            \n            inverse_unit : division unit_expression\n            \n            factor : factor_fits\n                   | factor_float\n                   | factor_int\n            \n            factor_float : signed_float\n                         | signed_float UINT signed_int\n                         | signed_float UINT power numeric_power\n            \n            factor_int : UINT\n                       | UINT signed_int\n                       | UINT power numeric_power\n                       | UINT UINT signed_int\n                       | UINT UINT power numeric_power\n            \n            factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN\n                        | UINT power OPEN_PAREN UINT CLOSE_PAREN\n                        | UINT power signed_int\n                        | UINT power UINT\n                        | UINT SIGN UINT\n                        | UINT OPEN_PAREN signed_int CLOSE_PAREN\n            \n            product_of_units : unit_expression product product_of_units\n                             | unit_expression product_of_units\n                             | unit_expression\n            \n            unit_expression : function\n                            | unit_with_power\n                            | OPEN_PAREN product_of_units CLOSE_PAREN\n            \n            unit_with_power : UNIT power numeric_power\n                            | UNIT numeric_power\n                            | UNIT\n            \n            numeric_power : sign UINT\n                          | OPEN_PAREN paren_expr CLOSE_PAREN\n            \n            paren_expr : sign UINT\n                       | signed_float\n                       | frac\n            \n            frac : sign UINT division sign UINT\n            \n            sign : SIGN\n                 |\n            \n            product : STAR\n                    | PERIOD\n            \n            division : SOLIDUS\n            \n            power : DOUBLE_STAR\n                  | CARET\n            \n            signed_int : SIGN UINT\n            \n            signed_float : sign UINT\n                         | sign UFLOAT\n            \n            function_name : FUNCNAME\n            \n            function : function_name OPEN_PAREN main CLOSE_PAREN\n            '
     
-_lr_action_items = {'OPEN_PAREN':([0,3,6,7,8,9,10,11,12,13,14,16,17,18,19,21,23,26,27,28,29,34,36,38,39,41,42,43,46,47,53,54,55,58,59,62,63,64,66,67,72,73,75,76,77,78,80,],[13,13,13,-14,-15,-16,13,-32,-33,13,35,-17,-48,41,45,-54,13,-46,-47,13,13,57,-21,-49,-50,13,45,-36,-52,-53,-34,-23,45,-26,-22,-27,-18,45,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'UINT':([0,14,15,16,17,19,20,34,37,38,39,41,42,44,45,46,47,55,56,57,60,64,69,81,82,],[14,33,-44,40,-48,-45,46,-45,62,-49,-50,14,-45,67,-45,-52,-53,-45,73,-45,73,-45,79,-45,83,]),'SOLIDUS':([0,2,3,4,6,7,8,9,11,12,14,16,19,22,23,24,26,27,30,36,41,43,46,47,48,49,51,52,53,54,58,59,62,63,66,67,72,73,75,76,77,78,79,80,],[17,-12,17,17,-31,-14,-15,-16,-32,-33,-20,-17,-37,-12,17,17,-46,-47,-30,-21,17,-36,-52,-53,-12,17,-11,-29,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,17,-25,]),'UNIT':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[19,19,19,-14,-15,-16,19,-32,-33,19,-20,-17,-48,-37,19,-46,-47,19,19,-21,19,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'FUNCNAME':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[21,21,21,-14,-15,-16,21,-32,-33,21,-20,-17,-48,-37,21,-46,-47,21,21,-21,21,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'SIGN':([0,14,17,19,33,34,35,38,39,40,41,42,45,55,57,64,81,],[15,37,-48,15,56,60,56,-49,-50,56,15,15,15,15,60,15,15,]),'UFLOAT':([0,15,20,41,45,57,60,69,],[-45,-44,47,-45,-45,-45,-44,47,]),'$end':([1,2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,36,43,46,47,48,49,50,51,52,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[0,-1,-10,-4,-7,-31,-14,-15,-16,-32,-33,-20,-17,-37,-2,-5,-8,-30,-13,-21,-36,-52,-53,-3,-6,-9,-11,-29,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'CLOSE_PAREN':([2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,32,36,43,46,47,48,49,50,51,52,53,54,58,59,61,62,63,65,66,67,68,70,71,72,73,74,75,76,77,78,79,80,83,],[-1,-10,-4,-7,-31,-14,-15,-16,-32,-33,-20,-17,-37,-2,-5,-8,-30,-13,53,-21,-36,-52,-53,-3,-6,-9,-11,-29,-34,-23,-26,-22,75,-27,-18,77,-35,-38,78,-41,-42,-24,-51,80,-28,-19,-55,-39,-40,-25,-43,]),'STAR':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[26,26,-14,-15,-16,-32,-33,-20,-17,-37,-21,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'PERIOD':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,58,59,62,63,66,67,72,73,75,76,77,78,80,],[27,27,-14,-15,-16,-32,-33,-20,-17,-37,-21,-36,-52,-53,-34,-23,-26,-22,-27,-18,-35,-38,-24,-51,-28,-19,-55,-39,-25,]),'DOUBLE_STAR':([14,19,33,40,],[38,38,38,38,]),'CARET':([14,19,33,40,],[39,39,39,39,]),}
+_lr_action_items = {'OPEN_PAREN':([0,3,6,7,8,9,10,11,12,13,14,16,17,18,19,21,23,26,27,28,29,34,36,38,39,41,42,43,46,47,53,54,55,57,59,60,63,64,65,67,68,73,74,77,78,79,80,82,83,],[13,13,13,-14,-15,-16,13,-34,-35,13,35,-17,-50,41,45,-56,13,-48,-49,13,13,58,-21,-51,-52,13,45,-38,-54,-55,-36,-23,45,-28,-27,-22,-29,-18,45,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'UINT':([0,14,15,16,17,19,20,34,37,38,39,41,42,44,45,46,47,55,56,58,61,65,70,84,85,],[14,33,-46,40,-50,-47,46,57,63,-51,-52,14,-47,68,-47,-54,-55,-47,74,75,74,-47,81,-47,86,]),'SOLIDUS':([0,2,3,4,6,7,8,9,11,12,14,16,19,22,23,24,26,27,30,36,41,43,46,47,48,49,51,52,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,81,82,83,],[17,-12,17,17,-33,-14,-15,-16,-34,-35,-20,-17,-39,-12,17,17,-48,-49,-32,-21,17,-38,-54,-55,-12,17,-11,-31,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,17,-26,-25,]),'UNIT':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[19,19,19,-14,-15,-16,19,-34,-35,19,-20,-17,-50,-39,19,-48,-49,19,19,-21,19,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'FUNCNAME':([0,3,6,7,8,9,10,11,12,13,14,16,17,19,23,26,27,28,29,36,41,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[21,21,21,-14,-15,-16,21,-34,-35,21,-20,-17,-50,-39,21,-48,-49,21,21,-21,21,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'SIGN':([0,14,17,19,33,34,35,38,39,40,41,42,45,55,58,65,84,],[15,37,-50,15,56,61,56,-51,-52,56,15,15,15,15,61,15,15,]),'UFLOAT':([0,15,20,41,45,58,61,70,],[-47,-46,47,-47,-47,-47,-46,47,]),'$end':([1,2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,36,43,46,47,48,49,50,51,52,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[0,-1,-10,-4,-7,-33,-14,-15,-16,-34,-35,-20,-17,-39,-2,-5,-8,-32,-13,-21,-38,-54,-55,-3,-6,-9,-11,-31,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'CLOSE_PAREN':([2,3,4,5,6,7,8,9,11,12,14,16,19,22,24,25,30,31,32,36,43,46,47,48,49,50,51,52,53,54,57,59,60,62,63,64,66,67,68,69,71,72,73,74,75,76,77,78,79,80,81,82,83,86,],[-1,-10,-4,-7,-33,-14,-15,-16,-34,-35,-20,-17,-39,-2,-5,-8,-32,-13,53,-21,-38,-54,-55,-3,-6,-9,-11,-31,-36,-23,-28,-27,-22,77,-29,-18,79,-37,-40,80,-43,-44,-24,-53,82,83,-30,-19,-57,-41,-42,-26,-25,-45,]),'STAR':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[26,26,-14,-15,-16,-34,-35,-20,-17,-39,-21,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'PERIOD':([3,6,7,8,9,11,12,14,16,19,36,43,46,47,53,54,57,59,60,63,64,67,68,73,74,77,78,79,80,82,83,],[27,27,-14,-15,-16,-34,-35,-20,-17,-39,-21,-38,-54,-55,-36,-23,-28,-27,-22,-29,-18,-37,-40,-24,-53,-30,-19,-57,-41,-26,-25,]),'DOUBLE_STAR':([14,19,33,40,],[38,38,38,38,]),'CARET':([14,19,33,40,],[39,39,39,39,]),}
 
 _lr_action = {}
 for _k, _v in _lr_action_items.items():
@@ -27,7 +27,7 @@
       _lr_action[_x][_k] = _y
 del _lr_action_items
 
-_lr_goto_items = {'main':([0,41,],[1,65,]),'product_of_units':([0,3,6,13,23,28,29,41,],[2,22,30,32,48,51,52,2,]),'factor':([0,41,],[3,3,]),'division_product_of_units':([0,3,23,41,],[4,24,49,4,]),'inverse_unit':([0,3,23,41,],[5,25,50,5,]),'unit_expression':([0,3,6,10,13,23,28,29,41,],[6,6,6,31,6,6,6,6,6,]),'factor_fits':([0,41,],[7,7,]),'factor_float':([0,41,],[8,8,]),'factor_int':([0,41,],[9,9,]),'division':([0,3,4,23,24,41,49,79,],[10,10,28,10,28,10,28,81,]),'function':([0,3,6,10,13,23,28,29,41,],[11,11,11,11,11,11,11,11,11,]),'unit_with_power':([0,3,6,10,13,23,28,29,41,],[12,12,12,12,12,12,12,12,12,]),'signed_float':([0,41,45,57,],[16,16,70,70,]),'function_name':([0,3,6,10,13,23,28,29,41,],[18,18,18,18,18,18,18,18,18,]),'sign':([0,19,34,41,42,45,55,57,64,81,],[20,44,44,20,44,69,44,69,44,82,]),'product':([3,6,],[23,29,]),'power':([14,19,33,40,],[34,42,55,64,]),'signed_int':([14,33,34,35,40,57,],[36,54,58,61,63,74,]),'numeric_power':([19,34,42,55,64,],[43,59,66,72,76,]),'paren_expr':([45,57,],[68,68,]),'frac':([45,57,],[71,71,]),}
+_lr_goto_items = {'main':([0,41,],[1,66,]),'product_of_units':([0,3,6,13,23,28,29,41,],[2,22,30,32,48,51,52,2,]),'factor':([0,41,],[3,3,]),'division_product_of_units':([0,3,23,41,],[4,24,49,4,]),'inverse_unit':([0,3,23,41,],[5,25,50,5,]),'unit_expression':([0,3,6,10,13,23,28,29,41,],[6,6,6,31,6,6,6,6,6,]),'factor_fits':([0,41,],[7,7,]),'factor_float':([0,41,],[8,8,]),'factor_int':([0,41,],[9,9,]),'division':([0,3,4,23,24,41,49,81,],[10,10,28,10,28,10,28,84,]),'function':([0,3,6,10,13,23,28,29,41,],[11,11,11,11,11,11,11,11,11,]),'unit_with_power':([0,3,6,10,13,23,28,29,41,],[12,12,12,12,12,12,12,12,12,]),'signed_float':([0,41,45,58,],[16,16,71,71,]),'function_name':([0,3,6,10,13,23,28,29,41,],[18,18,18,18,18,18,18,18,18,]),'sign':([0,19,34,41,42,45,55,58,65,84,],[20,44,44,20,44,70,44,70,44,85,]),'product':([3,6,],[23,29,]),'power':([14,19,33,40,],[34,42,55,65,]),'signed_int':([14,33,34,35,40,58,],[36,54,59,62,64,76,]),'numeric_power':([19,34,42,55,65,],[43,60,67,73,78,]),'paren_expr':([45,58,],[69,69,]),'frac':([45,58,],[72,72,]),}
 
 _lr_goto = {}
 for _k, _v in _lr_goto_items.items():
@@ -62,34 +62,36 @@
   ('factor_int -> UINT UINT signed_int','factor_int',3,'p_factor_int','generic.py',257),
   ('factor_int -> UINT UINT power numeric_power','factor_int',4,'p_factor_int','generic.py',258),
   ('factor_fits -> UINT power OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',5,'p_factor_fits','generic.py',276),
-  ('factor_fits -> UINT power signed_int','factor_fits',3,'p_factor_fits','generic.py',277),
-  ('factor_fits -> UINT SIGN UINT','factor_fits',3,'p_factor_fits','generic.py',278),
-  ('factor_fits -> UINT OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',4,'p_factor_fits','generic.py',279),
-  ('product_of_units -> unit_expression product product_of_units','product_of_units',3,'p_product_of_units','generic.py',298),
-  ('product_of_units -> unit_expression product_of_units','product_of_units',2,'p_product_of_units','generic.py',299),
-  ('product_of_units -> unit_expression','product_of_units',1,'p_product_of_units','generic.py',300),
-  ('unit_expression -> function','unit_expression',1,'p_unit_expression','generic.py',311),
-  ('unit_expression -> unit_with_power','unit_expression',1,'p_unit_expression','generic.py',312),
-  ('unit_expression -> OPEN_PAREN product_of_units CLOSE_PAREN','unit_expression',3,'p_unit_expression','generic.py',313),
-  ('unit_with_power -> UNIT power numeric_power','unit_with_power',3,'p_unit_with_power','generic.py',322),
-  ('unit_with_power -> UNIT numeric_power','unit_with_power',2,'p_unit_with_power','generic.py',323),
-  ('unit_with_power -> UNIT','unit_with_power',1,'p_unit_with_power','generic.py',324),
-  ('numeric_power -> sign UINT','numeric_power',2,'p_numeric_power','generic.py',335),
-  ('numeric_power -> OPEN_PAREN paren_expr CLOSE_PAREN','numeric_power',3,'p_numeric_power','generic.py',336),
-  ('paren_expr -> sign UINT','paren_expr',2,'p_paren_expr','generic.py',345),
-  ('paren_expr -> signed_float','paren_expr',1,'p_paren_expr','generic.py',346),
-  ('paren_expr -> frac','paren_expr',1,'p_paren_expr','generic.py',347),
-  ('frac -> sign UINT division sign UINT','frac',5,'p_frac','generic.py',356),
-  ('sign -> SIGN','sign',1,'p_sign','generic.py',362),
-  ('sign -> <empty>','sign',0,'p_sign','generic.py',363),
-  ('product -> STAR','product',1,'p_product','generic.py',372),
-  ('product -> PERIOD','product',1,'p_product','generic.py',373),
-  ('division -> SOLIDUS','division',1,'p_division','generic.py',379),
-  ('power -> DOUBLE_STAR','power',1,'p_power','generic.py',385),
-  ('power -> CARET','power',1,'p_power','generic.py',386),
-  ('signed_int -> SIGN UINT','signed_int',2,'p_signed_int','generic.py',392),
-  ('signed_float -> sign UINT','signed_float',2,'p_signed_float','generic.py',398),
-  ('signed_float -> sign UFLOAT','signed_float',2,'p_signed_float','generic.py',399),
-  ('function_name -> FUNCNAME','function_name',1,'p_function_name','generic.py',405),
-  ('function -> function_name OPEN_PAREN main CLOSE_PAREN','function',4,'p_function','generic.py',411),
+  ('factor_fits -> UINT power OPEN_PAREN UINT CLOSE_PAREN','factor_fits',5,'p_factor_fits','generic.py',277),
+  ('factor_fits -> UINT power signed_int','factor_fits',3,'p_factor_fits','generic.py',278),
+  ('factor_fits -> UINT power UINT','factor_fits',3,'p_factor_fits','generic.py',279),
+  ('factor_fits -> UINT SIGN UINT','factor_fits',3,'p_factor_fits','generic.py',280),
+  ('factor_fits -> UINT OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',4,'p_factor_fits','generic.py',281),
+  ('product_of_units -> unit_expression product product_of_units','product_of_units',3,'p_product_of_units','generic.py',300),
+  ('product_of_units -> unit_expression product_of_units','product_of_units',2,'p_product_of_units','generic.py',301),
+  ('product_of_units -> unit_expression','product_of_units',1,'p_product_of_units','generic.py',302),
+  ('unit_expression -> function','unit_expression',1,'p_unit_expression','generic.py',313),
+  ('unit_expression -> unit_with_power','unit_expression',1,'p_unit_expression','generic.py',314),
+  ('unit_expression -> OPEN_PAREN product_of_units CLOSE_PAREN','unit_expression',3,'p_unit_expression','generic.py',315),
+  ('unit_with_power -> UNIT power numeric_power','unit_with_power',3,'p_unit_with_power','generic.py',324),
+  ('unit_with_power -> UNIT numeric_power','unit_with_power',2,'p_unit_with_power','generic.py',325),
+  ('unit_with_power -> UNIT','unit_with_power',1,'p_unit_with_power','generic.py',326),
+  ('numeric_power -> sign UINT','numeric_power',2,'p_numeric_power','generic.py',337),
+  ('numeric_power -> OPEN_PAREN paren_expr CLOSE_PAREN','numeric_power',3,'p_numeric_power','generic.py',338),
+  ('paren_expr -> sign UINT','paren_expr',2,'p_paren_expr','generic.py',347),
+  ('paren_expr -> signed_float','paren_expr',1,'p_paren_expr','generic.py',348),
+  ('paren_expr -> frac','paren_expr',1,'p_paren_expr','generic.py',349),
+  ('frac -> sign UINT division sign UINT','frac',5,'p_frac','generic.py',358),
+  ('sign -> SIGN','sign',1,'p_sign','generic.py',364),
+  ('sign -> <empty>','sign',0,'p_sign','generic.py',365),
+  ('product -> STAR','product',1,'p_product','generic.py',374),
+  ('product -> PERIOD','product',1,'p_product','generic.py',375),
+  ('division -> SOLIDUS','division',1,'p_division','generic.py',381),
+  ('power -> DOUBLE_STAR','power',1,'p_power','generic.py',387),
+  ('power -> CARET','power',1,'p_power','generic.py',388),
+  ('signed_int -> SIGN UINT','signed_int',2,'p_signed_int','generic.py',394),
+  ('signed_float -> sign UINT','signed_float',2,'p_signed_float','generic.py',400),
+  ('signed_float -> sign UFLOAT','signed_float',2,'p_signed_float','generic.py',401),
+  ('function_name -> FUNCNAME','function_name',1,'p_function_name','generic.py',407),
+  ('function -> function_name OPEN_PAREN main CLOSE_PAREN','function',4,'p_function','generic.py',413),
 ]

```

## Test Patch

```diff
diff --git a/astropy/units/tests/test_format.py b/astropy/units/tests/test_format.py
--- a/astropy/units/tests/test_format.py
+++ b/astropy/units/tests/test_format.py
@@ -434,40 +434,42 @@ def test_vounit_implicit_custom():
     assert x.bases[0]._represents.bases[0].name == 'urlong'
 
 
-def test_fits_scale_factor():
+@pytest.mark.parametrize('scale, number, string',
+                         [('10+2', 100, '10**2'),
+                          ('10(+2)', 100, '10**2'),
+                          ('10**+2', 100, '10**2'),
+                          ('10**(+2)', 100, '10**2'),
+                          ('10^+2', 100, '10**2'),
+                          ('10^(+2)', 100, '10**2'),
+                          ('10**2', 100, '10**2'),
+                          ('10**(2)', 100, '10**2'),
+                          ('10^2', 100, '10**2'),
+                          ('10^(2)', 100, '10**2'),
+                          ('10-20', 10**(-20), '10**-20'),
+                          ('10(-20)', 10**(-20), '10**-20'),
+                          ('10**-20', 10**(-20), '10**-20'),
+                          ('10**(-20)', 10**(-20), '10**-20'),
+                          ('10^-20', 10**(-20), '10**-20'),
+                          ('10^(-20)', 10**(-20), '10**-20'),
+                          ])
+def test_fits_scale_factor(scale, number, string):
+
+    x = u.Unit(scale + ' erg/s/cm**2/Angstrom', format='fits')
+    assert x == number * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
+    assert x.to_string(format='fits') == string + ' Angstrom-1 cm-2 erg s-1'
+
+    x = u.Unit(scale + '*erg/s/cm**2/Angstrom', format='fits')
+    assert x == number * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
+    assert x.to_string(format='fits') == string + ' Angstrom-1 cm-2 erg s-1'
+
+
+def test_fits_scale_factor_errors():
     with pytest.raises(ValueError):
         x = u.Unit('1000 erg/s/cm**2/Angstrom', format='fits')
 
     with pytest.raises(ValueError):
         x = u.Unit('12 erg/s/cm**2/Angstrom', format='fits')
 
-    x = u.Unit('10+2 erg/s/cm**2/Angstrom', format='fits')
-    assert x == 100 * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
-    assert x.to_string(format='fits') == '10**2 Angstrom-1 cm-2 erg s-1'
-
-    x = u.Unit('10**(-20) erg/s/cm**2/Angstrom', format='fits')
-    assert x == 10**(-20) * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
-    assert x.to_string(format='fits') == '10**-20 Angstrom-1 cm-2 erg s-1'
-
-    x = u.Unit('10**-20 erg/s/cm**2/Angstrom', format='fits')
-    assert x == 10**(-20) * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
-    assert x.to_string(format='fits') == '10**-20 Angstrom-1 cm-2 erg s-1'
-
-    x = u.Unit('10^(-20) erg/s/cm**2/Angstrom', format='fits')
-    assert x == 10**(-20) * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
-    assert x.to_string(format='fits') == '10**-20 Angstrom-1 cm-2 erg s-1'
-
-    x = u.Unit('10^-20 erg/s/cm**2/Angstrom', format='fits')
-    assert x == 10**(-20) * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
-    assert x.to_string(format='fits') == '10**-20 Angstrom-1 cm-2 erg s-1'
-
-    x = u.Unit('10-20 erg/s/cm**2/Angstrom', format='fits')
-    assert x == 10**(-20) * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
-    assert x.to_string(format='fits') == '10**-20 Angstrom-1 cm-2 erg s-1'
-
-    x = u.Unit('10**(-20)*erg/s/cm**2/Angstrom', format='fits')
-    assert x == 10**(-20) * (u.erg / u.s / u.cm ** 2 / u.Angstrom)
-
     x = u.Unit(1.2 * u.erg)
     with pytest.raises(ValueError):
         x.to_string(format='fits')

```


## Code snippets

### 1 - astropy/units/format/fits.py:

Start line: 82, End line: 108

```python
class Fits(generic.Generic):

    @classmethod
    def _validate_unit(cls, unit, detailed_exception=True):
        if unit not in cls._units:
            if detailed_exception:
                raise ValueError(
                    "Unit '{0}' not supported by the FITS standard. {1}".format(
                        unit, utils.did_you_mean_units(
                            unit, cls._units, cls._deprecated_units,
                            cls._to_decomposed_alternative)))
            else:
                raise ValueError()

        if unit in cls._deprecated_units:
            utils.unit_deprecation_warning(
                unit, cls._units[unit], 'FITS',
                cls._to_decomposed_alternative)

    @classmethod
    def _parse_unit(cls, unit, detailed_exception=True):
        cls._validate_unit(unit)
        return cls._units[unit]

    @classmethod
    def _get_unit_name(cls, unit):
        name = unit.get_format_name('fits')
        cls._validate_unit(name)
        return name
```
### 2 - astropy/units/format/fits.py:

Start line: 139, End line: 158

```python
class Fits(generic.Generic):

    @classmethod
    def _to_decomposed_alternative(cls, unit):
        try:
            s = cls.to_string(unit)
        except core.UnitScaleError:
            scale = unit.scale
            unit = copy.copy(unit)
            unit._scale = 1.0
            return '{0} (with data multiplied by {1})'.format(
                cls.to_string(unit), scale)
        return s

    @classmethod
    def parse(cls, s, debug=False):
        result = super().parse(s, debug)
        if hasattr(result, 'function_unit'):
            raise ValueError("Function units are not yet supported for "
                             "FITS units.")
        return result
```
### 3 - astropy/units/format/fits.py:

Start line: 110, End line: 137

```python
class Fits(generic.Generic):

    @classmethod
    def to_string(cls, unit):
        # Remove units that aren't known to the format
        unit = utils.decompose_to_known_units(unit, cls._get_unit_name)

        parts = []

        if isinstance(unit, core.CompositeUnit):
            base = np.log10(unit.scale)

            if base % 1.0 != 0.0:
                raise core.UnitScaleError(
                    "The FITS unit format is not able to represent scales "
                    "that are not powers of 10.  Multiply your data by "
                    "{0:e}.".format(unit.scale))
            elif unit.scale != 1.0:
                parts.append('10**{0}'.format(int(base)))

            pairs = list(zip(unit.bases, unit.powers))
            if len(pairs):
                pairs.sort(key=operator.itemgetter(1), reverse=True)
                parts.append(cls._format_unit_list(pairs))

            s = ' '.join(parts)
        elif isinstance(unit, core.NamedUnit):
            s = cls._get_unit_name(unit)

        return s
```
### 4 - astropy/units/format/fits.py:

Start line: 8, End line: 80

```python
import numpy as np

import copy
import keyword
import operator

from . import core, generic, utils


class Fits(generic.Generic):
    """
    The FITS standard unit format.

    This supports the format defined in the Units section of the `FITS
    Standard <https://fits.gsfc.nasa.gov/fits_standard.html>`_.
    """

    name = 'fits'

    @staticmethod
    def _generate_unit_names():
        from astropy import units as u
        names = {}
        deprecated_names = set()

        # Note about deprecated units: before v2.0, several units were treated
        # as deprecated (G, barn, erg, Angstrom, angstrom). However, in the
        # FITS 3.0 standard, these units are explicitly listed in the allowed
        # units, but deprecated in the IAU Style Manual (McNally 1988). So
        # after discussion (https://github.com/astropy/astropy/issues/2933),
        # these units have been removed from the lists of deprecated units and
        # bases.

        bases = [
            'm', 'g', 's', 'rad', 'sr', 'K', 'A', 'mol', 'cd',
            'Hz', 'J', 'W', 'V', 'N', 'Pa', 'C', 'Ohm', 'S',
            'F', 'Wb', 'T', 'H', 'lm', 'lx', 'a', 'yr', 'eV',
            'pc', 'Jy', 'mag', 'R', 'bit', 'byte', 'G', 'barn'
        ]
        deprecated_bases = []
        prefixes = [
            'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', 'c', 'd',
            '', 'da', 'h', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']

        special_cases = {'dbyte': u.Unit('dbyte', 0.1*u.byte)}

        for base in bases + deprecated_bases:
            for prefix in prefixes:
                key = prefix + base
                if keyword.iskeyword(key):
                    continue
                elif key in special_cases:
                    names[key] = special_cases[key]
                else:
                    names[key] = getattr(u, key)
        for base in deprecated_bases:
            for prefix in prefixes:
                deprecated_names.add(prefix + base)

        simple_units = [
            'deg', 'arcmin', 'arcsec', 'mas', 'min', 'h', 'd', 'Ry',
            'solMass', 'u', 'solLum', 'solRad', 'AU', 'lyr', 'count',
            'ct', 'photon', 'ph', 'pixel', 'pix', 'D', 'Sun', 'chan',
            'bin', 'voxel', 'adu', 'beam', 'erg', 'Angstrom', 'angstrom'
        ]
        deprecated_units = []

        for unit in simple_units + deprecated_units:
            names[unit] = getattr(u, unit)
        for unit in deprecated_units:
            deprecated_names.add(unit)

        return names, deprecated_names, []
```
### 5 - astropy/units/astrophys.py:

Start line: 78, End line: 154

```python
def_unit(['M_e'], _si.m_e, namespace=_ns, doc="Electron mass",
         format={'latex': r'M_{e}', 'unicode': 'Mₑ'})
# Unified atomic mass unit
def_unit(['u', 'Da', 'Dalton'], _si.u, namespace=_ns,
         prefixes=True, exclude_prefixes=['a', 'da'],
         doc="Unified atomic mass unit")

##########################################################################
# ENERGY

# Here, explicitly convert the planck constant to 'eV s' since the constant
# can override that to give a more precise value that takes into account
# covariances between e and h.  Eventually, this may also be replaced with
# just `_si.Ryd.to(eV)`.
def_unit(['Ry', 'rydberg'],
         (_si.Ryd * _si.c * _si.h.to(si.eV * si.s)).to(si.eV),
         namespace=_ns, prefixes=True,
         doc="Rydberg: Energy of a photon whose wavenumber is the Rydberg "
         "constant",
         format={'latex': r'R_{\infty}', 'unicode': 'R∞'})


###########################################################################
# ILLUMINATION

def_unit(['solLum', 'L_sun', 'Lsun'], _si.L_sun, namespace=_ns,
         prefixes=False, doc="Solar luminance",
         format={'latex': r'L_{\odot}', 'unicode': 'L⊙'})


###########################################################################
# SPECTRAL DENSITY

def_unit((['ph', 'photon'], ['photon']),
         format={'ogip': 'photon', 'vounit': 'photon'},
         namespace=_ns, prefixes=True)
def_unit(['Jy', 'Jansky', 'jansky'], 1e-26 * si.W / si.m ** 2 / si.Hz,
         namespace=_ns, prefixes=True,
         doc="Jansky: spectral flux density")
def_unit(['R', 'Rayleigh', 'rayleigh'],
         (1e10 / (4 * _numpy.pi)) *
         ph * si.m ** -2 * si.s ** -1 * si.sr ** -1,
         namespace=_ns, prefixes=True,
         doc="Rayleigh: photon flux")


###########################################################################
# MISCELLANEOUS

# Some of these are very FITS-specific and perhaps considered a mistake.
# Maybe they should be moved into the FITS format class?
# TODO: This is defined by the FITS standard as "relative to the sun".
# Is that mass, volume, what?
def_unit(['Sun'], namespace=_ns)


###########################################################################
# EVENTS

def_unit((['ct', 'count'], ['count']),
         format={'fits': 'count', 'ogip': 'count', 'vounit': 'count'},
         namespace=_ns, prefixes=True, exclude_prefixes=['p'])
def_unit((['pix', 'pixel'], ['pixel']),
         format={'ogip': 'pixel', 'vounit': 'pixel'},
         namespace=_ns, prefixes=True)


###########################################################################
# MISCELLANEOUS

def_unit(['chan'], namespace=_ns, prefixes=True)
def_unit(['bin'], namespace=_ns, prefixes=True)
def_unit((['vox', 'voxel'], ['voxel']),
         format={'fits': 'voxel', 'ogip': 'voxel', 'vounit': 'voxel'},
         namespace=_ns, prefixes=True)
def_unit((['bit', 'b'], ['bit']), namespace=_ns,
         prefixes=si_prefixes + binary_prefixes)
```
### 6 - astropy/units/core.py:

Start line: 1693, End line: 1755

```python
class UnrecognizedUnit(IrreducibleUnit):
    """
    A unit that did not parse correctly.  This allows for
    round-tripping it as a string, but no unit operations actually work
    on it.

    Parameters
    ----------
    st : str
        The name of the unit.
    """
    # For UnrecognizedUnits, we want to use "standard" Python
    # pickling, not the special case that is used for
    # IrreducibleUnits.
    __reduce__ = object.__reduce__

    def __repr__(self):
        return "UnrecognizedUnit({0})".format(str(self))

    def __bytes__(self):
        return self.name.encode('ascii', 'replace')

    def __str__(self):
        return self.name

    def to_string(self, format=None):
        return self.name

    def _unrecognized_operator(self, *args, **kwargs):
        raise ValueError(
            "The unit {0!r} is unrecognized, so all arithmetic operations "
            "with it are invalid.".format(self.name))

    __pow__ = __div__ = __rdiv__ = __truediv__ = __rtruediv__ = __mul__ = \
        __rmul__ = __lt__ = __gt__ = __le__ = __ge__ = __neg__ = \
        _unrecognized_operator

    def __eq__(self, other):
        try:
            other = Unit(other, parse_strict='silent')
        except (ValueError, UnitsError, TypeError):
            return NotImplemented

        return isinstance(other, type(self)) and self.name == other.name

    def __ne__(self, other):
        return not (self == other)

    def is_equivalent(self, other, equivalencies=None):
        self._normalize_equivalencies(equivalencies)
        return self == other

    def _get_converter(self, other, equivalencies=None):
        self._normalize_equivalencies(equivalencies)
        raise ValueError(
            "The unit {0!r} is unrecognized.  It can not be converted "
            "to other units.".format(self.name))

    def get_format_name(self, format):
        return self.name

    def is_unity(self):
        return False
```
### 7 - astropy/units/format/generic.py:

Start line: 274, End line: 294

```python
class Generic(Base):

    @classmethod
    def _make_parser(cls):
        # ... other code

        def p_factor_fits(p):
            '''
            factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN
                        | UINT power signed_int
                        | UINT SIGN UINT
                        | UINT OPEN_PAREN signed_int CLOSE_PAREN
            '''
            if p[1] != 10:
                if cls.name == 'fits':
                    raise ValueError("Base must be 10")
                else:
                    return
            if len(p) == 4:
                if p[2] in ('**', '^'):
                    p[0] = 10 ** p[3]
                else:
                    p[0] = 10 ** (p[2] * p[3])
            elif len(p) == 5:
                p[0] = 10 ** p[3]
            elif len(p) == 6:
                p[0] = 10 ** p[4]
        # ... other code
```
### 8 - astropy/units/format/generic.py:

Start line: 440, End line: 495

```python
class Generic(Base):

    @classmethod
    def _get_unit(cls, t):
        try:
            return cls._parse_unit(t.value)
        except ValueError as e:
            raise ValueError(
                "At col {0}, {1}".format(
                    t.lexpos, str(e)))

    @classmethod
    def _parse_unit(cls, s, detailed_exception=True):
        registry = core.get_current_unit_registry().registry
        if s == '%':
            return registry['percent']
        elif s in registry:
            return registry[s]

        if detailed_exception:
            raise ValueError(
                '{0} is not a valid unit. {1}'.format(
                    s, did_you_mean(s, registry)))
        else:
            raise ValueError()

    @classmethod
    def parse(cls, s, debug=False):
        if not isinstance(s, str):
            s = s.decode('ascii')

        result = cls._do_parse(s, debug=debug)
        if s.count('/') > 1:
            warnings.warn(
                "'{0}' contains multiple slashes, which is "
                "discouraged by the FITS standard".format(s),
                core.UnitsWarning)
        return result

    @classmethod
    def _do_parse(cls, s, debug=False):
        try:
            # This is a short circuit for the case where the string
            # is just a single unit name
            return cls._parse_unit(s, detailed_exception=False)
        except ValueError as e:
            try:
                return cls._parser.parse(s, lexer=cls._lexer, debug=debug)
            except ValueError as e:
                if str(e):
                    raise
                else:
                    raise ValueError(
                        "Syntax error parsing unit '{0}'".format(s))

    @classmethod
    def _get_unit_name(cls, unit):
        return unit.get_format_name('generic')
```
### 9 - astropy/units/format/generic_parsetab.py:

Start line: 38, End line: 96

```python
_lr_productions = [
  ("S' -> main","S'",1,None,None,None),
  ('main -> product_of_units','main',1,'p_main','generic.py',193),
  ('main -> factor product_of_units','main',2,'p_main','generic.py',194),
  ('main -> factor product product_of_units','main',3,'p_main','generic.py',195),
  ('main -> division_product_of_units','main',1,'p_main','generic.py',196),
  ('main -> factor division_product_of_units','main',2,'p_main','generic.py',197),
  ('main -> factor product division_product_of_units','main',3,'p_main','generic.py',198),
  ('main -> inverse_unit','main',1,'p_main','generic.py',199),
  ('main -> factor inverse_unit','main',2,'p_main','generic.py',200),
  ('main -> factor product inverse_unit','main',3,'p_main','generic.py',201),
  ('main -> factor','main',1,'p_main','generic.py',202),
  ('division_product_of_units -> division_product_of_units division product_of_units','division_product_of_units',3,'p_division_product_of_units','generic.py',214),
  ('division_product_of_units -> product_of_units','division_product_of_units',1,'p_division_product_of_units','generic.py',215),
  ('inverse_unit -> division unit_expression','inverse_unit',2,'p_inverse_unit','generic.py',225),
  ('factor -> factor_fits','factor',1,'p_factor','generic.py',231),
  ('factor -> factor_float','factor',1,'p_factor','generic.py',232),
  ('factor -> factor_int','factor',1,'p_factor','generic.py',233),
  ('factor_float -> signed_float','factor_float',1,'p_factor_float','generic.py',239),
  ('factor_float -> signed_float UINT signed_int','factor_float',3,'p_factor_float','generic.py',240),
  ('factor_float -> signed_float UINT power numeric_power','factor_float',4,'p_factor_float','generic.py',241),
  ('factor_int -> UINT','factor_int',1,'p_factor_int','generic.py',254),
  ('factor_int -> UINT signed_int','factor_int',2,'p_factor_int','generic.py',255),
  ('factor_int -> UINT power numeric_power','factor_int',3,'p_factor_int','generic.py',256),
  ('factor_int -> UINT UINT signed_int','factor_int',3,'p_factor_int','generic.py',257),
  ('factor_int -> UINT UINT power numeric_power','factor_int',4,'p_factor_int','generic.py',258),
  ('factor_fits -> UINT power OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',5,'p_factor_fits','generic.py',276),
  ('factor_fits -> UINT power signed_int','factor_fits',3,'p_factor_fits','generic.py',277),
  ('factor_fits -> UINT SIGN UINT','factor_fits',3,'p_factor_fits','generic.py',278),
  ('factor_fits -> UINT OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',4,'p_factor_fits','generic.py',279),
  ('product_of_units -> unit_expression product product_of_units','product_of_units',3,'p_product_of_units','generic.py',298),
  ('product_of_units -> unit_expression product_of_units','product_of_units',2,'p_product_of_units','generic.py',299),
  ('product_of_units -> unit_expression','product_of_units',1,'p_product_of_units','generic.py',300),
  ('unit_expression -> function','unit_expression',1,'p_unit_expression','generic.py',311),
  ('unit_expression -> unit_with_power','unit_expression',1,'p_unit_expression','generic.py',312),
  ('unit_expression -> OPEN_PAREN product_of_units CLOSE_PAREN','unit_expression',3,'p_unit_expression','generic.py',313),
  ('unit_with_power -> UNIT power numeric_power','unit_with_power',3,'p_unit_with_power','generic.py',322),
  ('unit_with_power -> UNIT numeric_power','unit_with_power',2,'p_unit_with_power','generic.py',323),
  ('unit_with_power -> UNIT','unit_with_power',1,'p_unit_with_power','generic.py',324),
  ('numeric_power -> sign UINT','numeric_power',2,'p_numeric_power','generic.py',335),
  ('numeric_power -> OPEN_PAREN paren_expr CLOSE_PAREN','numeric_power',3,'p_numeric_power','generic.py',336),
  ('paren_expr -> sign UINT','paren_expr',2,'p_paren_expr','generic.py',345),
  ('paren_expr -> signed_float','paren_expr',1,'p_paren_expr','generic.py',346),
  ('paren_expr -> frac','paren_expr',1,'p_paren_expr','generic.py',347),
  ('frac -> sign UINT division sign UINT','frac',5,'p_frac','generic.py',356),
  ('sign -> SIGN','sign',1,'p_sign','generic.py',362),
  ('sign -> <empty>','sign',0,'p_sign','generic.py',363),
  ('product -> STAR','product',1,'p_product','generic.py',372),
  ('product -> PERIOD','product',1,'p_product','generic.py',373),
  ('division -> SOLIDUS','division',1,'p_division','generic.py',379),
  ('power -> DOUBLE_STAR','power',1,'p_power','generic.py',385),
  ('power -> CARET','power',1,'p_power','generic.py',386),
  ('signed_int -> SIGN UINT','signed_int',2,'p_signed_int','generic.py',392),
  ('signed_float -> sign UINT','signed_float',2,'p_signed_float','generic.py',398),
  ('signed_float -> sign UFLOAT','signed_float',2,'p_signed_float','generic.py',399),
  ('function_name -> FUNCNAME','function_name',1,'p_function_name','generic.py',405),
  ('function -> function_name OPEN_PAREN main CLOSE_PAREN','function',4,'p_function','generic.py',411),
]
```
### 10 - astropy/units/quantity_helper/erfa.py:

Start line: 1, End line: 43

```python
# -*- coding: utf-8 -*-


from astropy.units.core import UnitsError, UnitTypeError, dimensionless_unscaled
from . import UFUNC_HELPERS
from .helpers import get_converter, helper_invariant, helper_multiplication


erfa_ufuncs = ('s2c', 's2p', 'c2s', 'p2s', 'pm', 'pdp', 'pxp', 'rxp')


def helper_s2c(f, unit1, unit2):
    from astropy.units.si import radian
    try:
        return [get_converter(unit1, radian),
                get_converter(unit2, radian)], dimensionless_unscaled
    except UnitsError:
        raise UnitTypeError("Can only apply '{0}' function to "
                            "quantities with angle units"
                            .format(f.__name__))


def helper_s2p(f, unit1, unit2, unit3):
    from astropy.units.si import radian
    try:
        return [get_converter(unit1, radian),
                get_converter(unit2, radian), None], unit3
    except UnitsError:
        raise UnitTypeError("Can only apply '{0}' function to "
                            "quantities with angle units"
                            .format(f.__name__))


def helper_c2s(f, unit1):
    from astropy.units.si import radian
    return [None], (radian, radian)


def helper_p2s(f, unit1):
    from astropy.units.si import radian
    return [None], (radian, radian, unit1)
```
### 12 - astropy/units/format/generic.py:

Start line: 252, End line: 272

```python
class Generic(Base):

    @classmethod
    def _make_parser(cls):
        # ... other code

        def p_factor_int(p):
            '''
            factor_int : UINT
                       | UINT signed_int
                       | UINT power numeric_power
                       | UINT UINT signed_int
                       | UINT UINT power numeric_power
            '''
            if cls.name == 'fits':
                raise ValueError("Numeric factor not supported by FITS")
            if len(p) == 2:
                p[0] = p[1]
            elif len(p) == 3:
                p[0] = p[1] ** float(p[2])
            elif len(p) == 4:
                if isinstance(p[2], int):
                    p[0] = p[1] * p[2] ** float(p[3])
                else:
                    p[0] = p[1] ** float(p[3])
            elif len(p) == 5:
                p[0] = p[1] * p[2] ** p[4]
        # ... other code
```
### 16 - astropy/units/format/generic.py:

Start line: 55, End line: 103

```python
class Generic(Base):
    """
    A "generic" format.

    The syntax of the format is based directly on the FITS standard,
    but instead of only supporting the units that FITS knows about, it
    supports any unit available in the `astropy.units` namespace.
    """

    _show_scale = True

    _tokens = (
        'DOUBLE_STAR',
        'STAR',
        'PERIOD',
        'SOLIDUS',
        'CARET',
        'OPEN_PAREN',
        'CLOSE_PAREN',
        'FUNCNAME',
        'UNIT',
        'SIGN',
        'UINT',
        'UFLOAT'
    )

    @classproperty(lazy=True)
    def _all_units(cls):
        return cls._generate_unit_names()

    @classproperty(lazy=True)
    def _units(cls):
        return cls._all_units[0]

    @classproperty(lazy=True)
    def _deprecated_units(cls):
        return cls._all_units[1]

    @classproperty(lazy=True)
    def _functions(cls):
        return cls._all_units[2]

    @classproperty(lazy=True)
    def _parser(cls):
        return cls._make_parser()

    @classproperty(lazy=True)
    def _lexer(cls):
        return cls._make_lexer()
```
### 19 - astropy/units/format/generic.py:

Start line: 237, End line: 250

```python
class Generic(Base):

    @classmethod
    def _make_parser(cls):
        # ... other code

        def p_factor_float(p):
            '''
            factor_float : signed_float
                         | signed_float UINT signed_int
                         | signed_float UINT power numeric_power
            '''
            if cls.name == 'fits':
                raise ValueError("Numeric factor not supported by FITS")
            if len(p) == 4:
                p[0] = p[1] * p[2] ** float(p[3])
            elif len(p) == 5:
                p[0] = p[1] * p[2] ** float(p[4])
            elif len(p) == 2:
                p[0] = p[1]
        # ... other code
```
### 36 - astropy/units/format/generic_parsetab.py:

Start line: 1, End line: 21

```python
# -*- coding: utf-8 -*-
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'DOUBLE_STAR STAR PERIOD SOLIDUS CARET OPEN_PAREN CLOSE_PAREN FUNCNAME UNIT SIGN UINT UFLOAT\n            main : product_of_units\n                 | factor product_of_units\n                 | factor product product_of_units\n                 | division_product_of_units\n                 | factor division_product_of_units\n                 | factor product division_product_of_units\n                 | inverse_unit\n                 | factor inverse_unit\n                 | factor product inverse_unit\n                 | factor\n            \n            division_product_of_units : division_product_of_units division product_of_units\n                                      | product_of_units\n            \n            inverse_unit : division unit_expression\n            \n            factor : factor_fits\n                   | factor_float\n                   | factor_int\n            \n            factor_float : signed_float\n                         | signed_float UINT signed_int\n                         | signed_float UINT power numeric_power\n            \n            factor_int : UINT\n                       | UINT signed_int\n                       | UINT power numeric_power\n                       | UINT UINT signed_int\n                       | UINT UINT power numeric_power\n            \n            factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN\n                        | UINT power signed_int\n                        | UINT SIGN UINT\n                        | UINT OPEN_PAREN signed_int CLOSE_PAREN\n            \n            product_of_units : unit_expression product product_of_units\n                             | unit_expression product_of_units\n                             | unit_expression\n            \n            unit_expression : function\n                            | unit_with_power\n                            | OPEN_PAREN product_of_units CLOSE_PAREN\n            \n            unit_with_power : UNIT power numeric_power\n                            | UNIT numeric_power\n                            | UNIT\n            \n            numeric_power : sign UINT\n                          | OPEN_PAREN paren_expr CLOSE_PAREN\n            \n            paren_expr : sign UINT\n                       | signed_float\n                       | frac\n            \n            frac : sign UINT division sign UINT\n            \n            sign : SIGN\n                 |\n            \n            product : STAR\n                    | PERIOD\n            \n            division : SOLIDUS\n            \n            power : DOUBLE_STAR\n                  | CARET\n            \n            signed_int : SIGN UINT\n            \n            signed_float : sign UINT\n                         | sign UFLOAT\n            \n            function_name : FUNCNAME\n            \n            function : function_name OPEN_PAREN main CLOSE_PAREN\n            '

_lr_action_items =
 # ... other code
```
### 71 - astropy/units/format/generic.py:

Start line: 191, End line: 210

```python
class Generic(Base):

    @classmethod
    def _make_parser(cls):
        # ... other code

        def p_main(p):
            '''
            main : product_of_units
                 | factor product_of_units
                 | factor product product_of_units
                 | division_product_of_units
                 | factor division_product_of_units
                 | factor product division_product_of_units
                 | inverse_unit
                 | factor inverse_unit
                 | factor product inverse_unit
                 | factor
            '''
            from astropy.units.core import Unit
            if len(p) == 2:
                p[0] = Unit(p[1])
            elif len(p) == 3:
                p[0] = Unit(p[1] * p[2])
            elif len(p) == 4:
                p[0] = Unit(p[1] * p[3])
        # ... other code
```
### 90 - astropy/units/format/generic.py:

Start line: 497, End line: 528

```python
class Generic(Base):

    @classmethod
    def _format_unit_list(cls, units):
        out = []
        units.sort(key=lambda x: cls._get_unit_name(x[0]).lower())

        for base, power in units:
            if power == 1:
                out.append(cls._get_unit_name(base))
            else:
                power = utils.format_power(power)
                if '/' in power:
                    out.append('{0}({1})'.format(
                        cls._get_unit_name(base), power))
                else:
                    out.append('{0}{1}'.format(
                        cls._get_unit_name(base), power))
        return ' '.join(out)

    @classmethod
    def to_string(cls, unit):
        return _to_string(cls, unit)


class Unscaled(Generic):
    """
    A format that doesn't display the scale part of the unit, other
    than that, it is identical to the `Generic` format.

    This is used in some error messages where the scale is irrelevant.
    """
    _show_scale = False
```
### 101 - astropy/units/format/generic.py:

Start line: 296, End line: 407

```python
class Generic(Base):

    @classmethod
    def _make_parser(cls):
        # ... other code

        def p_product_of_units(p):
            '''
            product_of_units : unit_expression product product_of_units
                             | unit_expression product_of_units
                             | unit_expression
            '''
            if len(p) == 2:
                p[0] = p[1]
            elif len(p) == 3:
                p[0] = p[1] * p[2]
            else:
                p[0] = p[1] * p[3]

        def p_unit_expression(p):
            '''
            unit_expression : function
                            | unit_with_power
                            | OPEN_PAREN product_of_units CLOSE_PAREN
            '''
            if len(p) == 2:
                p[0] = p[1]
            else:
                p[0] = p[2]

        def p_unit_with_power(p):
            '''
            unit_with_power : UNIT power numeric_power
                            | UNIT numeric_power
                            | UNIT
            '''
            if len(p) == 2:
                p[0] = p[1]
            elif len(p) == 3:
                p[0] = p[1] ** p[2]
            else:
                p[0] = p[1] ** p[3]

        def p_numeric_power(p):
            '''
            numeric_power : sign UINT
                          | OPEN_PAREN paren_expr CLOSE_PAREN
            '''
            if len(p) == 3:
                p[0] = p[1] * p[2]
            elif len(p) == 4:
                p[0] = p[2]

        def p_paren_expr(p):
            '''
            paren_expr : sign UINT
                       | signed_float
                       | frac
            '''
            if len(p) == 3:
                p[0] = p[1] * p[2]
            else:
                p[0] = p[1]

        def p_frac(p):
            '''
            frac : sign UINT division sign UINT
            '''
            p[0] = (p[1] * p[2]) / (p[4] * p[5])

        def p_sign(p):
            '''
            sign : SIGN
                 |
            '''
            if len(p) == 2:
                p[0] = p[1]
            else:
                p[0] = 1.0

        def p_product(p):
            '''
            product : STAR
                    | PERIOD
            '''
            pass

        def p_division(p):
            '''
            division : SOLIDUS
            '''
            pass

        def p_power(p):
            '''
            power : DOUBLE_STAR
                  | CARET
            '''
            p[0] = p[1]

        def p_signed_int(p):
            '''
            signed_int : SIGN UINT
            '''
            p[0] = p[1] * p[2]

        def p_signed_float(p):
            '''
            signed_float : sign UINT
                         | sign UFLOAT
            '''
            p[0] = p[1] * p[2]

        def p_function_name(p):
            '''
            function_name : FUNCNAME
            '''
            p[0] = p[1]
        # ... other code
```
### 106 - astropy/units/format/generic.py:

Start line: 173, End line: 189

```python
class Generic(Base):

    @classmethod
    def _make_parser(cls):
        """
        The grammar here is based on the description in the `FITS
        standard
        <http://fits.gsfc.nasa.gov/standard30/fits_standard30aa.pdf>`_,
        Section 4.3, which is not terribly precise.  The exact grammar
        is here is based on the YACC grammar in the `unity library
        <https://bitbucket.org/nxg/unity/>`_.

        This same grammar is used by the `"fits"` and `"vounit"`
        formats, the only difference being the set of available unit
        strings.
        """
        from astropy.extern.ply import yacc

        tokens = cls._tokens
        # ... other code
```
### 107 - astropy/units/format/generic.py:

Start line: 212, End line: 235

```python
class Generic(Base):

    @classmethod
    def _make_parser(cls):
        # ... other code

        def p_division_product_of_units(p):
            '''
            division_product_of_units : division_product_of_units division product_of_units
                                      | product_of_units
            '''
            from astropy.units.core import Unit
            if len(p) == 4:
                p[0] = Unit(p[1] / p[3])
            else:
                p[0] = p[1]

        def p_inverse_unit(p):
            '''
            inverse_unit : division unit_expression
            '''
            p[0] = p[2] ** -1

        def p_factor(p):
            '''
            factor : factor_fits
                   | factor_float
                   | factor_int
            '''
            p[0] = p[1]
        # ... other code
```
