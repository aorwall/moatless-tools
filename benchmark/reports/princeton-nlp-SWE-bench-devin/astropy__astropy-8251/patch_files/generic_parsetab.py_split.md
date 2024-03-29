

## EpicSplitter

3 chunks

#### Split 1
523 tokens, line: 1 - 21

```python
# -*- coding: utf-8 -*-
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'DOUBLE_STAR STAR PERIOD SOLIDUS CARET OPEN_PAREN CLOSE_PAREN FUNCNAME UNIT SIGN UINT UFLOAT\n            main : product_of_units\n                 | factor product_of_units\n                 | factor product product_of_units\n                 | division_product_of_units\n                 | factor division_product_of_units\n                 | factor product division_product_of_units\n                 | inverse_unit\n                 | factor inverse_unit\n                 | factor product inverse_unit\n                 | factor\n            \n            division_product_of_units : division_product_of_units division product_of_units\n                                      | product_of_units\n            \n            inverse_unit : division unit_expression\n            \n            factor : factor_fits\n                   | factor_float\n                   | factor_int\n            \n            factor_float : signed_float\n                         | signed_float UINT signed_int\n                         | signed_float UINT power numeric_power\n            \n            factor_int : UINT\n                       | UINT signed_int\n                       | UINT power numeric_power\n                       | UINT UINT signed_int\n                       | UINT UINT power numeric_power\n            \n            factor_fits : UINT power OPEN_PAREN signed_int CLOSE_PAREN\n                        | UINT power signed_int\n                        | UINT SIGN UINT\n                        | UINT OPEN_PAREN signed_int CLOSE_PAREN\n            \n            product_of_units : unit_expression product product_of_units\n                             | unit_expression product_of_units\n                             | unit_expression\n            \n            unit_expression : function\n                            | unit_with_power\n                            | OPEN_PAREN product_of_units CLOSE_PAREN\n            \n            unit_with_power : UNIT power numeric_power\n                            | UNIT numeric_power\n                            | UNIT\n            \n            numeric_power : sign UINT\n                          | OPEN_PAREN paren_expr CLOSE_PAREN\n            \n            paren_expr : sign UINT\n                       | signed_float\n                       | frac\n            \n            frac : sign UINT division sign UINT\n            \n            sign : SIGN\n                 |\n            \n            product : STAR\n                    | PERIOD\n            \n            division : SOLIDUS\n            \n            power : DOUBLE_STAR\n                  | CARET\n            \n            signed_int : SIGN UINT\n            \n            signed_float : sign UINT\n                         | sign UFLOAT\n            \n            function_name : FUNCNAME\n            \n            function : function_name OPEN_PAREN main CLOSE_PAREN\n            '

_lr_action_items =
 # ... other code
```



#### Split 2
714 tokens, line: 23 - 37

```python
_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'main':([0,41,],[1,65,]),'product_of_units':([0,3,6,13,23,28,29,41,],[2,22,30,32,48,51,52,2,]),'factor':([0,41,],[3,3,]),'division_product_of_units':([0,3,23,41,],[4,24,49,4,]),'inverse_unit':([0,3,23,41,],[5,25,50,5,]),'unit_expression':([0,3,6,10,13,23,28,29,41,],[6,6,6,31,6,6,6,6,6,]),'factor_fits':([0,41,],[7,7,]),'factor_float':([0,41,],[8,8,]),'factor_int':([0,41,],[9,9,]),'division':([0,3,4,23,24,41,49,79,],[10,10,28,10,28,10,28,81,]),'function':([0,3,6,10,13,23,28,29,41,],[11,11,11,11,11,11,11,11,11,]),'unit_with_power':([0,3,6,10,13,23,28,29,41,],[12,12,12,12,12,12,12,12,12,]),'signed_float':([0,41,45,57,],[16,16,70,70,]),'function_name':([0,3,6,10,13,23,28,29,41,],[18,18,18,18,18,18,18,18,18,]),'sign':([0,19,34,41,42,45,55,57,64,81,],[20,44,44,20,44,69,44,69,44,82,]),'product':([3,6,],[23,29,]),'power':([14,19,33,40,],[34,42,55,64,]),'signed_int':([14,33,34,35,40,57,],[36,54,58,61,63,74,]),'numeric_power':([19,34,42,55,64,],[43,59,66,72,76,]),'paren_expr':([45,57,],[68,68,]),'frac':([45,57,],[71,71,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
```



#### Split 3
1303 tokens, line: 38 - 96

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

