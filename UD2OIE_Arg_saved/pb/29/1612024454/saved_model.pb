дц

я2¬2
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
>
DiagPart

input"T
diagonal"T"
Ttype:

2	
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
≠
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
°
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
.
Identity

input"T
output"T"	
Ttype
…
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0ю€€€€€€€€"
value_indexint(0ю€€€€€€€€"+

vocab_sizeint€€€€€€€€€(0€€€€€€€€€"
	delimiterstring	И
А
LSTMBlockCell
x"T
cs_prev"T
h_prev"T
w"T
wci"T
wcf"T
wco"T
b"T
i"T
cs"T
f"T
o"T
ci"T
co"T
h"T"
forget_biasfloat%  А?"
	cell_clipfloat%  @@"
use_peepholebool( "
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
Р
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
TouttypeИ
!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
М
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
Q
Qr

input"T
q"T
r"T"
full_matricesbool( "
Ttype:	
2
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
К
ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sign
x"T
y"T"
Ttype:

2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:И
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeИ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
TtypeИ
9
TensorArraySizeV3

handle
flow_in
sizeИ
ё
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring И
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
TtypeИ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.14.02unknown8жА	

global_step/Initializer/zerosConst*
_output_shapes
: *
value	B	 R *
_class
loc:@global_step*
dtype0	
k
global_step
VariableV2*
shape: *
_output_shapes
: *
_class
loc:@global_step*
dtype0	
Й
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_output_shapes
: *
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
~
	arg_inputPlaceholder*%
shape:€€€€€€€€€€€€€€€€€€*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
dtype0
}
arg_maskPlaceholder*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
dtype0*%
shape:€€€€€€€€€€€€€€€€€€
x
arg_original_sequence_lengthsPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
|
arg_depPlaceholder*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
dtype0*%
shape:€€€€€€€€€€€€€€€€€€
~
	prop_maskPlaceholder*%
shape:€€€€€€€€€€€€€€€€€€*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
dtype0
s
"Arguments/pos/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
e
 Arguments/pos/random_uniform/minConst*
_output_shapes
: *
valueB
 *  Ањ*
dtype0
e
 Arguments/pos/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
Х
*Arguments/pos/random_uniform/RandomUniformRandomUniform"Arguments/pos/random_uniform/shape*
_output_shapes

:*
dtype0*
T0
М
 Arguments/pos/random_uniform/subSub Arguments/pos/random_uniform/max Arguments/pos/random_uniform/min*
_output_shapes
: *
T0
Ю
 Arguments/pos/random_uniform/mulMul*Arguments/pos/random_uniform/RandomUniform Arguments/pos/random_uniform/sub*
_output_shapes

:*
T0
Р
Arguments/pos/random_uniformAdd Arguments/pos/random_uniform/mul Arguments/pos/random_uniform/min*
_output_shapes

:*
T0
f
Arguments/pos/Variable
VariableV2*
shape
:*
_output_shapes

:*
dtype0
±
Arguments/pos/Variable/AssignAssignArguments/pos/VariableArguments/pos/random_uniform*)
_class
loc:@Arguments/pos/Variable*
T0*
_output_shapes

:
У
Arguments/pos/Variable/readIdentityArguments/pos/Variable*)
_class
loc:@Arguments/pos/Variable*
T0*
_output_shapes

:
Р
#Arguments/pos/embedding_lookup/axisConst*
value	B : *)
_class
loc:@Arguments/pos/Variable*
dtype0*
_output_shapes
: 
Д
Arguments/pos/embedding_lookupGatherV2Arguments/pos/Variable/read	arg_input#Arguments/pos/embedding_lookup/axis*)
_class
loc:@Arguments/pos/Variable*
Tindices0*
Taxis0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
Tparams0
Т
'Arguments/pos/embedding_lookup/IdentityIdentityArguments/pos/embedding_lookup*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
s
"Arguments/dep/random_uniform/shapeConst*
valueB">      *
dtype0*
_output_shapes
:
e
 Arguments/dep/random_uniform/minConst*
valueB
 *  Ањ*
dtype0*
_output_shapes
: 
e
 Arguments/dep/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Х
*Arguments/dep/random_uniform/RandomUniformRandomUniform"Arguments/dep/random_uniform/shape*
dtype0*
T0*
_output_shapes

:>
М
 Arguments/dep/random_uniform/subSub Arguments/dep/random_uniform/max Arguments/dep/random_uniform/min*
T0*
_output_shapes
: 
Ю
 Arguments/dep/random_uniform/mulMul*Arguments/dep/random_uniform/RandomUniform Arguments/dep/random_uniform/sub*
T0*
_output_shapes

:>
Р
Arguments/dep/random_uniformAdd Arguments/dep/random_uniform/mul Arguments/dep/random_uniform/min*
T0*
_output_shapes

:>
f
Arguments/dep/Variable
VariableV2*
dtype0*
shape
:>*
_output_shapes

:>
±
Arguments/dep/Variable/AssignAssignArguments/dep/VariableArguments/dep/random_uniform*)
_class
loc:@Arguments/dep/Variable*
T0*
_output_shapes

:>
У
Arguments/dep/Variable/readIdentityArguments/dep/Variable*)
_class
loc:@Arguments/dep/Variable*
T0*
_output_shapes

:>
Р
#Arguments/dep/embedding_lookup/axisConst*
value	B : *)
_class
loc:@Arguments/dep/Variable*
dtype0*
_output_shapes
: 
В
Arguments/dep/embedding_lookupGatherV2Arguments/dep/Variable/readarg_dep#Arguments/dep/embedding_lookup/axis*)
_class
loc:@Arguments/dep/Variable*
Taxis0*
Tindices0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
Tparams0
Т
'Arguments/dep/embedding_lookup/IdentityIdentityArguments/dep/embedding_lookup*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
x
'Arguments/arg_mask/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
j
%Arguments/arg_mask/random_uniform/minConst*
dtype0*
valueB
 *  Ањ*
_output_shapes
: 
j
%Arguments/arg_mask/random_uniform/maxConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Я
/Arguments/arg_mask/random_uniform/RandomUniformRandomUniform'Arguments/arg_mask/random_uniform/shape*
dtype0*
T0*
_output_shapes

:
Ы
%Arguments/arg_mask/random_uniform/subSub%Arguments/arg_mask/random_uniform/max%Arguments/arg_mask/random_uniform/min*
T0*
_output_shapes
: 
≠
%Arguments/arg_mask/random_uniform/mulMul/Arguments/arg_mask/random_uniform/RandomUniform%Arguments/arg_mask/random_uniform/sub*
T0*
_output_shapes

:
Я
!Arguments/arg_mask/random_uniformAdd%Arguments/arg_mask/random_uniform/mul%Arguments/arg_mask/random_uniform/min*
T0*
_output_shapes

:
k
Arguments/arg_mask/Variable
VariableV2*
dtype0*
shape
:*
_output_shapes

:
≈
"Arguments/arg_mask/Variable/AssignAssignArguments/arg_mask/Variable!Arguments/arg_mask/random_uniform*.
_class$
" loc:@Arguments/arg_mask/Variable*
T0*
_output_shapes

:
Ґ
 Arguments/arg_mask/Variable/readIdentityArguments/arg_mask/Variable*
T0*.
_class$
" loc:@Arguments/arg_mask/Variable*
_output_shapes

:
Ъ
(Arguments/arg_mask/embedding_lookup/axisConst*
value	B : *
dtype0*.
_class$
" loc:@Arguments/arg_mask/Variable*
_output_shapes
: 
Ч
#Arguments/arg_mask/embedding_lookupGatherV2 Arguments/arg_mask/Variable/readarg_mask(Arguments/arg_mask/embedding_lookup/axis*
Tindices0*
Tparams0*
Taxis0*.
_class$
" loc:@Arguments/arg_mask/Variable*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Ь
,Arguments/arg_mask/embedding_lookup/IdentityIdentity#Arguments/arg_mask/embedding_lookup*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
y
(Arguments/prop_mask/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
k
&Arguments/prop_mask/random_uniform/minConst*
valueB
 *  Ањ*
dtype0*
_output_shapes
: 
k
&Arguments/prop_mask/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
°
0Arguments/prop_mask/random_uniform/RandomUniformRandomUniform(Arguments/prop_mask/random_uniform/shape*
dtype0*
T0*
_output_shapes

:
Ю
&Arguments/prop_mask/random_uniform/subSub&Arguments/prop_mask/random_uniform/max&Arguments/prop_mask/random_uniform/min*
_output_shapes
: *
T0
∞
&Arguments/prop_mask/random_uniform/mulMul0Arguments/prop_mask/random_uniform/RandomUniform&Arguments/prop_mask/random_uniform/sub*
_output_shapes

:*
T0
Ґ
"Arguments/prop_mask/random_uniformAdd&Arguments/prop_mask/random_uniform/mul&Arguments/prop_mask/random_uniform/min*
_output_shapes

:*
T0
l
Arguments/prop_mask/Variable
VariableV2*
_output_shapes

:*
shape
:*
dtype0
…
#Arguments/prop_mask/Variable/AssignAssignArguments/prop_mask/Variable"Arguments/prop_mask/random_uniform*
_output_shapes

:*
T0*/
_class%
#!loc:@Arguments/prop_mask/Variable
•
!Arguments/prop_mask/Variable/readIdentityArguments/prop_mask/Variable*
_output_shapes

:*
T0*/
_class%
#!loc:@Arguments/prop_mask/Variable
Ь
)Arguments/prop_mask/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *
dtype0*/
_class%
#!loc:@Arguments/prop_mask/Variable
Ь
$Arguments/prop_mask/embedding_lookupGatherV2!Arguments/prop_mask/Variable/read	prop_mask)Arguments/prop_mask/embedding_lookup/axis*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
Tparams0*
Tindices0*
Taxis0*/
_class%
#!loc:@Arguments/prop_mask/Variable
Ю
-Arguments/prop_mask/embedding_lookup/IdentityIdentity$Arguments/prop_mask/embedding_lookup*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
T0
`
Arguments/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Ќ
Arguments/concatConcatV2'Arguments/pos/embedding_lookup/Identity'Arguments/dep/embedding_lookup/IdentityArguments/concat/axis*
T0*
N*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€(
b
Arguments/concat_1/axisConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
№
Arguments/concat_1ConcatV2,Arguments/arg_mask/embedding_lookup/Identity-Arguments/prop_mask/embedding_lookup/IdentityArguments/concat_1/axis*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€(*
N
b
Arguments/concat_2/axisConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
•
Arguments/concat_2ConcatV2Arguments/concatArguments/concat_1Arguments/concat_2/axis*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€P*
N
Б
$Arguments/index_to_string/asset_pathConst"/device:CPU:**
dtype0*
_output_shapes
: *
valueB Bindex_tag.txt
c
Arguments/index_to_string/ConstConst*
dtype0*
_output_shapes
: *
valueB	 BUNK
†
$Arguments/index_to_string/hash_tableHashTableV2*
value_dtype0*/
shared_name hash_table_index_tag.txt_-1_-2*
_output_shapes
: *
	key_dtype0	
е
BArguments/index_to_string/table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2$Arguments/index_to_string/hash_table$Arguments/index_to_string/asset_path*
	key_index€€€€€€€€€*
value_indexю€€€€€€€€
v
1Arguments/BiLSTM/forward/DropoutWrapperInit/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
x
3Arguments/BiLSTM/forward/DropoutWrapperInit/Const_1Const*
dtype0*
valueB
 *  А?*
_output_shapes
: 
x
3Arguments/BiLSTM/forward/DropoutWrapperInit/Const_2Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
(Arguments/BiLSTM/forward/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
s
(Arguments/BiLSTM/forward/concat/values_1Const*
valueB:»*
dtype0*
_output_shapes
:
f
$Arguments/BiLSTM/forward/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
”
Arguments/BiLSTM/forward/concatConcatV2(Arguments/BiLSTM/forward/concat/values_0(Arguments/BiLSTM/forward/concat/values_1$Arguments/BiLSTM/forward/concat/axis*
T0*
N*
_output_shapes
:
p
+Arguments/BiLSTM/forward/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+Arguments/BiLSTM/forward/random_uniform/maxConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Ю
5Arguments/BiLSTM/forward/random_uniform/RandomUniformRandomUniformArguments/BiLSTM/forward/concat*
dtype0*
T0*
_output_shapes
:	»
≠
+Arguments/BiLSTM/forward/random_uniform/subSub+Arguments/BiLSTM/forward/random_uniform/max+Arguments/BiLSTM/forward/random_uniform/min*
T0*
_output_shapes
: 
ј
+Arguments/BiLSTM/forward/random_uniform/mulMul5Arguments/BiLSTM/forward/random_uniform/RandomUniform+Arguments/BiLSTM/forward/random_uniform/sub*
T0*
_output_shapes
:	»
≤
'Arguments/BiLSTM/forward/random_uniformAdd+Arguments/BiLSTM/forward/random_uniform/mul+Arguments/BiLSTM/forward/random_uniform/min*
T0*
_output_shapes
:	»
t
*Arguments/BiLSTM/forward/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
u
*Arguments/BiLSTM/forward/concat_1/values_1Const*
dtype0*
_output_shapes
:*
valueB:»
h
&Arguments/BiLSTM/forward/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
џ
!Arguments/BiLSTM/forward/concat_1ConcatV2*Arguments/BiLSTM/forward/concat_1/values_0*Arguments/BiLSTM/forward/concat_1/values_1&Arguments/BiLSTM/forward/concat_1/axis*
T0*
_output_shapes
:*
N
r
-Arguments/BiLSTM/forward/random_uniform_1/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
r
-Arguments/BiLSTM/forward/random_uniform_1/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
7Arguments/BiLSTM/forward/random_uniform_1/RandomUniformRandomUniform!Arguments/BiLSTM/forward/concat_1*
dtype0*
T0*
_output_shapes
:	»
≥
-Arguments/BiLSTM/forward/random_uniform_1/subSub-Arguments/BiLSTM/forward/random_uniform_1/max-Arguments/BiLSTM/forward/random_uniform_1/min*
T0*
_output_shapes
: 
∆
-Arguments/BiLSTM/forward/random_uniform_1/mulMul7Arguments/BiLSTM/forward/random_uniform_1/RandomUniform-Arguments/BiLSTM/forward/random_uniform_1/sub*
T0*
_output_shapes
:	»
Є
)Arguments/BiLSTM/forward/random_uniform_1Add-Arguments/BiLSTM/forward/random_uniform_1/mul-Arguments/BiLSTM/forward/random_uniform_1/min*
T0*
_output_shapes
:	»
t
*Arguments/BiLSTM/forward/concat_2/values_0Const*
valueB:*
dtype0*
_output_shapes
:
u
*Arguments/BiLSTM/forward/concat_2/values_1Const*
valueB:»*
dtype0*
_output_shapes
:
h
&Arguments/BiLSTM/forward/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
џ
!Arguments/BiLSTM/forward/concat_2ConcatV2*Arguments/BiLSTM/forward/concat_2/values_0*Arguments/BiLSTM/forward/concat_2/values_1&Arguments/BiLSTM/forward/concat_2/axis*
T0*
N*
_output_shapes
:
r
-Arguments/BiLSTM/forward/random_uniform_2/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-Arguments/BiLSTM/forward/random_uniform_2/maxConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Ґ
7Arguments/BiLSTM/forward/random_uniform_2/RandomUniformRandomUniform!Arguments/BiLSTM/forward/concat_2*
dtype0*
T0*
_output_shapes
:	»
≥
-Arguments/BiLSTM/forward/random_uniform_2/subSub-Arguments/BiLSTM/forward/random_uniform_2/max-Arguments/BiLSTM/forward/random_uniform_2/min*
T0*
_output_shapes
: 
∆
-Arguments/BiLSTM/forward/random_uniform_2/mulMul7Arguments/BiLSTM/forward/random_uniform_2/RandomUniform-Arguments/BiLSTM/forward/random_uniform_2/sub*
T0*
_output_shapes
:	»
Є
)Arguments/BiLSTM/forward/random_uniform_2Add-Arguments/BiLSTM/forward/random_uniform_2/mul-Arguments/BiLSTM/forward/random_uniform_2/min*
T0*
_output_shapes
:	»
w
2Arguments/BiLSTM/backward/DropoutWrapperInit/ConstConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
y
4Arguments/BiLSTM/backward/DropoutWrapperInit/Const_1Const*
_output_shapes
: *
valueB
 *  А?*
dtype0
y
4Arguments/BiLSTM/backward/DropoutWrapperInit/Const_2Const*
_output_shapes
: *
valueB
 *  А?*
dtype0
s
)Arguments/BiLSTM/backward/concat/values_0Const*
_output_shapes
:*
valueB:*
dtype0
t
)Arguments/BiLSTM/backward/concat/values_1Const*
_output_shapes
:*
valueB:»*
dtype0
g
%Arguments/BiLSTM/backward/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
„
 Arguments/BiLSTM/backward/concatConcatV2)Arguments/BiLSTM/backward/concat/values_0)Arguments/BiLSTM/backward/concat/values_1%Arguments/BiLSTM/backward/concat/axis*
T0*
_output_shapes
:*
N
q
,Arguments/BiLSTM/backward/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
q
,Arguments/BiLSTM/backward/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
†
6Arguments/BiLSTM/backward/random_uniform/RandomUniformRandomUniform Arguments/BiLSTM/backward/concat*
_output_shapes
:	»*
dtype0*
T0
∞
,Arguments/BiLSTM/backward/random_uniform/subSub,Arguments/BiLSTM/backward/random_uniform/max,Arguments/BiLSTM/backward/random_uniform/min*
_output_shapes
: *
T0
√
,Arguments/BiLSTM/backward/random_uniform/mulMul6Arguments/BiLSTM/backward/random_uniform/RandomUniform,Arguments/BiLSTM/backward/random_uniform/sub*
_output_shapes
:	»*
T0
µ
(Arguments/BiLSTM/backward/random_uniformAdd,Arguments/BiLSTM/backward/random_uniform/mul,Arguments/BiLSTM/backward/random_uniform/min*
_output_shapes
:	»*
T0
u
+Arguments/BiLSTM/backward/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
v
+Arguments/BiLSTM/backward/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:»
i
'Arguments/BiLSTM/backward/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
я
"Arguments/BiLSTM/backward/concat_1ConcatV2+Arguments/BiLSTM/backward/concat_1/values_0+Arguments/BiLSTM/backward/concat_1/values_1'Arguments/BiLSTM/backward/concat_1/axis*
N*
_output_shapes
:*
T0
s
.Arguments/BiLSTM/backward/random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
s
.Arguments/BiLSTM/backward/random_uniform_1/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
8Arguments/BiLSTM/backward/random_uniform_1/RandomUniformRandomUniform"Arguments/BiLSTM/backward/concat_1*
T0*
_output_shapes
:	»*
dtype0
ґ
.Arguments/BiLSTM/backward/random_uniform_1/subSub.Arguments/BiLSTM/backward/random_uniform_1/max.Arguments/BiLSTM/backward/random_uniform_1/min*
T0*
_output_shapes
: 
…
.Arguments/BiLSTM/backward/random_uniform_1/mulMul8Arguments/BiLSTM/backward/random_uniform_1/RandomUniform.Arguments/BiLSTM/backward/random_uniform_1/sub*
T0*
_output_shapes
:	»
ї
*Arguments/BiLSTM/backward/random_uniform_1Add.Arguments/BiLSTM/backward/random_uniform_1/mul.Arguments/BiLSTM/backward/random_uniform_1/min*
T0*
_output_shapes
:	»
u
+Arguments/BiLSTM/backward/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB:
v
+Arguments/BiLSTM/backward/concat_2/values_1Const*
dtype0*
_output_shapes
:*
valueB:»
i
'Arguments/BiLSTM/backward/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
я
"Arguments/BiLSTM/backward/concat_2ConcatV2+Arguments/BiLSTM/backward/concat_2/values_0+Arguments/BiLSTM/backward/concat_2/values_1'Arguments/BiLSTM/backward/concat_2/axis*
T0*
_output_shapes
:*
N
s
.Arguments/BiLSTM/backward/random_uniform_2/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
s
.Arguments/BiLSTM/backward/random_uniform_2/maxConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
§
8Arguments/BiLSTM/backward/random_uniform_2/RandomUniformRandomUniform"Arguments/BiLSTM/backward/concat_2*
T0*
_output_shapes
:	»*
dtype0
ґ
.Arguments/BiLSTM/backward/random_uniform_2/subSub.Arguments/BiLSTM/backward/random_uniform_2/max.Arguments/BiLSTM/backward/random_uniform_2/min*
T0*
_output_shapes
: 
…
.Arguments/BiLSTM/backward/random_uniform_2/mulMul8Arguments/BiLSTM/backward/random_uniform_2/RandomUniform.Arguments/BiLSTM/backward/random_uniform_2/sub*
T0*
_output_shapes
:	»
ї
*Arguments/BiLSTM/backward/random_uniform_2Add.Arguments/BiLSTM/backward/random_uniform_2/mul.Arguments/BiLSTM/backward/random_uniform_2/min*
T0*
_output_shapes
:	»
d
"Arguments/BiLSTM/BiLSTM/fw/fw/RankConst*
_output_shapes
: *
value	B :*
dtype0
k
)Arguments/BiLSTM/BiLSTM/fw/fw/range/startConst*
_output_shapes
: *
dtype0*
value	B :
k
)Arguments/BiLSTM/BiLSTM/fw/fw/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
¬
#Arguments/BiLSTM/BiLSTM/fw/fw/rangeRange)Arguments/BiLSTM/BiLSTM/fw/fw/range/start"Arguments/BiLSTM/BiLSTM/fw/fw/Rank)Arguments/BiLSTM/BiLSTM/fw/fw/range/delta*
_output_shapes
:
~
-Arguments/BiLSTM/BiLSTM/fw/fw/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
k
)Arguments/BiLSTM/BiLSTM/fw/fw/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ё
$Arguments/BiLSTM/BiLSTM/fw/fw/concatConcatV2-Arguments/BiLSTM/BiLSTM/fw/fw/concat/values_0#Arguments/BiLSTM/BiLSTM/fw/fw/range)Arguments/BiLSTM/BiLSTM/fw/fw/concat/axis*
N*
_output_shapes
:*
T0
≠
'Arguments/BiLSTM/BiLSTM/fw/fw/transpose	TransposeArguments/concat_2$Arguments/BiLSTM/BiLSTM/fw/fw/concat*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€P*
T0
Ж
-Arguments/BiLSTM/BiLSTM/fw/fw/sequence_lengthIdentityarg_original_sequence_lengths*#
_output_shapes
:€€€€€€€€€*
T0
z
#Arguments/BiLSTM/BiLSTM/fw/fw/ShapeShape'Arguments/BiLSTM/BiLSTM/fw/fw/transpose*
_output_shapes
:*
T0
{
1Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
}
3Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
}
3Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
√
+Arguments/BiLSTM/BiLSTM/fw/fw/strided_sliceStridedSlice#Arguments/BiLSTM/BiLSTM/fw/fw/Shape1Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice/stack3Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_13Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
Э
[Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
Ф
WArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims
ExpandDims+Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice[Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dim*
_output_shapes
:*
T0
Э
RArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstConst*
_output_shapes
:*
valueB:»*
dtype0
Ъ
XArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ф
SArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatConcatV2WArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDimsRArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstXArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axis*
_output_shapes
:*
T0*
N
Э
XArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Љ
RArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zerosFillSArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatXArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/Const*(
_output_shapes
:€€€€€€€€€»*
T0
Я
]Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dimConst*
_output_shapes
: *
value	B : *
dtype0
Ш
YArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1
ExpandDims+Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice]Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:
Я
TArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_1Const*
_output_shapes
:*
valueB:»*
dtype0
Я
]Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dimConst*
_output_shapes
: *
value	B : *
dtype0
Ш
YArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2
ExpandDims+Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice]Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dim*
T0*
_output_shapes
:
Я
TArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2Const*
_output_shapes
:*
valueB:»*
dtype0
Ь
ZArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ь
UArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1ConcatV2YArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2TArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2ZArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axis*
T0*
_output_shapes
:*
N
Я
ZArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
¬
TArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1FillUArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1ZArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/Const*
T0*(
_output_shapes
:€€€€€€€€€»
Я
]Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dimConst*
_output_shapes
: *
value	B : *
dtype0
Ш
YArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3
ExpandDims+Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice]Arguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:
Я
TArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_3Const*
_output_shapes
:*
valueB:»*
dtype0
В
%Arguments/BiLSTM/BiLSTM/fw/fw/Shape_1Shape-Arguments/BiLSTM/BiLSTM/fw/fw/sequence_length*
T0*
_output_shapes
:
Ж
#Arguments/BiLSTM/BiLSTM/fw/fw/stackPack+Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice*
N*
_output_shapes
:*
T0
Э
#Arguments/BiLSTM/BiLSTM/fw/fw/EqualEqual%Arguments/BiLSTM/BiLSTM/fw/fw/Shape_1#Arguments/BiLSTM/BiLSTM/fw/fw/stack*
T0*
_output_shapes
:
m
#Arguments/BiLSTM/BiLSTM/fw/fw/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
К
!Arguments/BiLSTM/BiLSTM/fw/fw/AllAll#Arguments/BiLSTM/BiLSTM/fw/fw/Equal#Arguments/BiLSTM/BiLSTM/fw/fw/Const*
_output_shapes
: 
Є
*Arguments/BiLSTM/BiLSTM/fw/fw/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMExpected shape for Tensor Arguments/BiLSTM/BiLSTM/fw/fw/sequence_length:0 is 
}
,Arguments/BiLSTM/BiLSTM/fw/fw/Assert/Const_1Const*
_output_shapes
: *
dtype0*!
valueB B but saw shape: 
ј
2Arguments/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMExpected shape for Tensor Arguments/BiLSTM/BiLSTM/fw/fw/sequence_length:0 is 
Г
2Arguments/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*!
valueB B but saw shape: 
Ы
+Arguments/BiLSTM/BiLSTM/fw/fw/Assert/AssertAssert!Arguments/BiLSTM/BiLSTM/fw/fw/All2Arguments/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_0#Arguments/BiLSTM/BiLSTM/fw/fw/stack2Arguments/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_2%Arguments/BiLSTM/BiLSTM/fw/fw/Shape_1*
T
2
ј
)Arguments/BiLSTM/BiLSTM/fw/fw/CheckSeqLenIdentity-Arguments/BiLSTM/BiLSTM/fw/fw/sequence_length,^Arguments/BiLSTM/BiLSTM/fw/fw/Assert/Assert*#
_output_shapes
:€€€€€€€€€*
T0
|
%Arguments/BiLSTM/BiLSTM/fw/fw/Shape_2Shape'Arguments/BiLSTM/BiLSTM/fw/fw/transpose*
T0*
_output_shapes
:
}
3Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:

5Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
-Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1StridedSlice%Arguments/BiLSTM/BiLSTM/fw/fw/Shape_23Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack5Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_15Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
|
%Arguments/BiLSTM/BiLSTM/fw/fw/Shape_3Shape'Arguments/BiLSTM/BiLSTM/fw/fw/transpose*
T0*
_output_shapes
:
}
3Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:

5Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ќ
-Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2StridedSlice%Arguments/BiLSTM/BiLSTM/fw/fw/Shape_33Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack5Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_15Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
n
,Arguments/BiLSTM/BiLSTM/fw/fw/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Є
(Arguments/BiLSTM/BiLSTM/fw/fw/ExpandDims
ExpandDims-Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_2,Arguments/BiLSTM/BiLSTM/fw/fw/ExpandDims/dim*
T0*
_output_shapes
:
p
%Arguments/BiLSTM/BiLSTM/fw/fw/Const_1Const*
dtype0*
valueB:»*
_output_shapes
:
m
+Arguments/BiLSTM/BiLSTM/fw/fw/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ё
&Arguments/BiLSTM/BiLSTM/fw/fw/concat_1ConcatV2(Arguments/BiLSTM/BiLSTM/fw/fw/ExpandDims%Arguments/BiLSTM/BiLSTM/fw/fw/Const_1+Arguments/BiLSTM/BiLSTM/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:
n
)Arguments/BiLSTM/BiLSTM/fw/fw/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
±
#Arguments/BiLSTM/BiLSTM/fw/fw/zerosFill&Arguments/BiLSTM/BiLSTM/fw/fw/concat_1)Arguments/BiLSTM/BiLSTM/fw/fw/zeros/Const*
T0*(
_output_shapes
:€€€€€€€€€»
o
%Arguments/BiLSTM/BiLSTM/fw/fw/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
Ы
!Arguments/BiLSTM/BiLSTM/fw/fw/MinMin)Arguments/BiLSTM/BiLSTM/fw/fw/CheckSeqLen%Arguments/BiLSTM/BiLSTM/fw/fw/Const_2*
T0*
_output_shapes
: 
o
%Arguments/BiLSTM/BiLSTM/fw/fw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
Ы
!Arguments/BiLSTM/BiLSTM/fw/fw/MaxMax)Arguments/BiLSTM/BiLSTM/fw/fw/CheckSeqLen%Arguments/BiLSTM/BiLSTM/fw/fw/Const_3*
T0*
_output_shapes
: 
d
"Arguments/BiLSTM/BiLSTM/fw/fw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
¶
)Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayTensorArrayV3-Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1*
identical_element_shapes(*I
tensor_array_name42Arguments/BiLSTM/BiLSTM/fw/fw/dynamic_rnn/output_0*
dtype0*%
element_shape:€€€€€€€€€»*
_output_shapes

:: 
¶
+Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray_1TensorArrayV3-Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1*
identical_element_shapes(*H
tensor_array_name31Arguments/BiLSTM/BiLSTM/fw/fw/dynamic_rnn/input_0*
dtype0*$
element_shape:€€€€€€€€€P*
_output_shapes

:: 
Н
6Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/ShapeShape'Arguments/BiLSTM/BiLSTM/fw/fw/transpose*
T0*
_output_shapes
:
О
DArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Р
FArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Р
FArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ґ
>Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice6Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/ShapeDArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stackFArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_1FArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
~
<Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
~
<Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
†
6Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/rangeRange<Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/start>Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice<Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:€€€€€€€€€
К
XArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3+Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray_16Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range'Arguments/BiLSTM/BiLSTM/fw/fw/transpose-Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray_1:1*:
_class0
.,loc:@Arguments/BiLSTM/BiLSTM/fw/fw/transpose*
T0*
_output_shapes
: 
i
'Arguments/BiLSTM/BiLSTM/fw/fw/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B :
Э
%Arguments/BiLSTM/BiLSTM/fw/fw/MaximumMaximum'Arguments/BiLSTM/BiLSTM/fw/fw/Maximum/x!Arguments/BiLSTM/BiLSTM/fw/fw/Max*
_output_shapes
: *
T0
І
%Arguments/BiLSTM/BiLSTM/fw/fw/MinimumMinimum-Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1%Arguments/BiLSTM/BiLSTM/fw/fw/Maximum*
_output_shapes
: *
T0
w
5Arguments/BiLSTM/BiLSTM/fw/fw/while/iteration_counterConst*
_output_shapes
: *
dtype0*
value	B : 
и
)Arguments/BiLSTM/BiLSTM/fw/fw/while/EnterEnter5Arguments/BiLSTM/BiLSTM/fw/fw/while/iteration_counter*
_output_shapes
: *A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0*
parallel_iterations 
„
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_1Enter"Arguments/BiLSTM/BiLSTM/fw/fw/time*
_output_shapes
: *A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0*
parallel_iterations 
а
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_2Enter+Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray:1*
_output_shapes
: *A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0*
parallel_iterations 
Щ
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_3EnterRArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros*(
_output_shapes
:€€€€€€€€€»*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0*
parallel_iterations 
Ы
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_4EnterTArguments/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1*(
_output_shapes
:€€€€€€€€€»*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0*
parallel_iterations 
Љ
)Arguments/BiLSTM/BiLSTM/fw/fw/while/MergeMerge)Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter1Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration*
T0*
_output_shapes
: : *
N
¬
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_1Merge+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_13Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_1*
T0*
_output_shapes
: : *
N
¬
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_2Merge+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_23Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_2*
T0*
_output_shapes
: : *
N
‘
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_3Merge+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_33Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_3*
T0**
_output_shapes
:€€€€€€€€€»: *
N
‘
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_4Merge+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_43Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_4*
T0**
_output_shapes
:€€€€€€€€€»: *
N
ђ
(Arguments/BiLSTM/BiLSTM/fw/fw/while/LessLess)Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge.Arguments/BiLSTM/BiLSTM/fw/fw/while/Less/Enter*
T0*
_output_shapes
: 
ш
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Less/EnterEnter-Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1*
is_constant(*
T0*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
_output_shapes
: *
parallel_iterations 
≤
*Arguments/BiLSTM/BiLSTM/fw/fw/while/Less_1Less+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_10Arguments/BiLSTM/BiLSTM/fw/fw/while/Less_1/Enter*
T0*
_output_shapes
: 
т
0Arguments/BiLSTM/BiLSTM/fw/fw/while/Less_1/EnterEnter%Arguments/BiLSTM/BiLSTM/fw/fw/Minimum*
is_constant(*
T0*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
_output_shapes
: *
parallel_iterations 
™
.Arguments/BiLSTM/BiLSTM/fw/fw/while/LogicalAnd
LogicalAnd(Arguments/BiLSTM/BiLSTM/fw/fw/while/Less*Arguments/BiLSTM/BiLSTM/fw/fw/while/Less_1*
_output_shapes
: 
А
,Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCondLoopCond.Arguments/BiLSTM/BiLSTM/fw/fw/while/LogicalAnd*
_output_shapes
: 
о
*Arguments/BiLSTM/BiLSTM/fw/fw/while/SwitchSwitch)Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge,Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCond*
T0*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge*
_output_shapes
: : 
ф
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_1Switch+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_1,Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_1*
_output_shapes
: : 
ф
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_2Switch+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_2,Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_2*
_output_shapes
: : 
Ш
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_3Switch+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_3,Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_3*<
_output_shapes*
(:€€€€€€€€€»:€€€€€€€€€»
Ш
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_4Switch+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_4,Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_4*<
_output_shapes*
(:€€€€€€€€€»:€€€€€€€€€»
З
,Arguments/BiLSTM/BiLSTM/fw/fw/while/IdentityIdentity,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch:1*
T0*
_output_shapes
: 
Л
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_1Identity.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 
Л
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_2Identity.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 
Э
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_3Identity.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_3:1*
T0*(
_output_shapes
:€€€€€€€€€»
Э
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_4Identity.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_4:1*
T0*(
_output_shapes
:€€€€€€€€€»
Ъ
)Arguments/BiLSTM/BiLSTM/fw/fw/while/add/yConst-^Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
®
'Arguments/BiLSTM/BiLSTM/fw/fw/while/addAdd,Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity)Arguments/BiLSTM/BiLSTM/fw/fw/while/add/y*
_output_shapes
: *
T0
ђ
5Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3TensorArrayReadV3;Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_1=Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:€€€€€€€€€P*
dtype0
З
;Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/EnterEnter+Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray_1*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
_output_shapes
:*
is_constant(*
T0*
parallel_iterations 
≤
=Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1EnterXArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
_output_shapes
: *
is_constant(*
T0*
parallel_iterations 
÷
0Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqualGreaterEqual.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_16Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Enter*#
_output_shapes
:€€€€€€€€€*
T0
Й
6Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/EnterEnter)Arguments/BiLSTM/BiLSTM/fw/fw/CheckSeqLen*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*#
_output_shapes
:€€€€€€€€€*
is_constant(*
T0*
parallel_iterations 
ѕ
EArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"     *
_output_shapes
:*
dtype0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
Ѕ
CArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *¶Шљ*
_output_shapes
: *
dtype0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
Ѕ
CArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *¶Ш=*
_output_shapes
: *
dtype0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
Ц
MArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformEArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Ш†*
dtype0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel*
T0
Ѓ
CArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/subSubCArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/maxCArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
¬
CArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/mulMulMArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformCArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
Ш†*
T0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
і
?Arguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniformAddCArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/mulCArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
Ш†*
T0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
±
$Arguments/BiLSTM/fw/lstm_cell/kernel
VariableV2* 
_output_shapes
:
Ш†*
shape:
Ш†*
dtype0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
А
+Arguments/BiLSTM/fw/lstm_cell/kernel/AssignAssign$Arguments/BiLSTM/fw/lstm_cell/kernel?Arguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
Ш†*
T0*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel
Ж
)Arguments/BiLSTM/fw/lstm_cell/kernel/readIdentity$Arguments/BiLSTM/fw/lstm_cell/kernel* 
_output_shapes
:
Ш†*
T0
Ї
4Arguments/BiLSTM/fw/lstm_cell/bias/Initializer/ConstConst*
valueB†*    *
_output_shapes	
:†*
dtype0*5
_class+
)'loc:@Arguments/BiLSTM/fw/lstm_cell/bias
£
"Arguments/BiLSTM/fw/lstm_cell/bias
VariableV2*
shape:†*
_output_shapes	
:†*
dtype0*5
_class+
)'loc:@Arguments/BiLSTM/fw/lstm_cell/bias
к
)Arguments/BiLSTM/fw/lstm_cell/bias/AssignAssign"Arguments/BiLSTM/fw/lstm_cell/bias4Arguments/BiLSTM/fw/lstm_cell/bias/Initializer/Const*5
_class+
)'loc:@Arguments/BiLSTM/fw/lstm_cell/bias*
_output_shapes	
:†*
T0
}
'Arguments/BiLSTM/fw/lstm_cell/bias/readIdentity"Arguments/BiLSTM/fw/lstm_cell/bias*
_output_shapes	
:†*
T0
±
3Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zerosConst-^Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity*
valueB»*    *
dtype0*
_output_shapes	
:»
ћ
;Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCellLSTMBlockCell5Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_3.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_4AArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter3Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zeros3Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zeros3Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zerosCArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1*
	cell_clip%  Ањ*Ґ
_output_shapesП
М:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»*
T0
С
AArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/EnterEnter)Arguments/BiLSTM/fw/lstm_cell/kernel/read*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context* 
_output_shapes
:
Ш†*
is_constant(*
T0*
parallel_iterations 
М
CArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1Enter'Arguments/BiLSTM/fw/lstm_cell/bias/read*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
_output_shapes	
:†*
is_constant(*
T0*
parallel_iterations 
Џ
*Arguments/BiLSTM/BiLSTM/fw/fw/while/SelectSelect0Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual0Arguments/BiLSTM/BiLSTM/fw/fw/while/Select/Enter=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:6*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*(
_output_shapes
:€€€€€€€€€»*
T0
“
0Arguments/BiLSTM/BiLSTM/fw/fw/while/Select/EnterEnter#Arguments/BiLSTM/BiLSTM/fw/fw/zeros*
is_constant(*A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context*
parallel_iterations *N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*(
_output_shapes
:€€€€€€€€€»*
T0
Џ
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Select_1Select0Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_3=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:1*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*(
_output_shapes
:€€€€€€€€€»*
T0
Џ
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Select_2Select0Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_4=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:6*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*(
_output_shapes
:€€€€€€€€€»*
T0
©
GArguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3MArguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_1*Arguments/BiLSTM/BiLSTM/fw/fw/while/Select.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_2*
_output_shapes
: *N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
T0
з
MArguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter)Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
T0*
_output_shapes
:*
is_constant(*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context
Ь
+Arguments/BiLSTM/BiLSTM/fw/fw/while/add_1/yConst-^Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ѓ
)Arguments/BiLSTM/BiLSTM/fw/fw/while/add_1Add.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_1+Arguments/BiLSTM/BiLSTM/fw/fw/while/add_1/y*
_output_shapes
: *
T0
М
1Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIterationNextIteration'Arguments/BiLSTM/BiLSTM/fw/fw/while/add*
_output_shapes
: *
T0
Р
3Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_1NextIteration)Arguments/BiLSTM/BiLSTM/fw/fw/while/add_1*
T0*
_output_shapes
: 
Ѓ
3Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_2NextIterationGArguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
•
3Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_3NextIteration,Arguments/BiLSTM/BiLSTM/fw/fw/while/Select_1*
T0*(
_output_shapes
:€€€€€€€€€»
•
3Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_4NextIteration,Arguments/BiLSTM/BiLSTM/fw/fw/while/Select_2*
T0*(
_output_shapes
:€€€€€€€€€»
}
(Arguments/BiLSTM/BiLSTM/fw/fw/while/ExitExit*Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch*
T0*
_output_shapes
: 
Б
*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_1Exit,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
Б
*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_2Exit,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_2*
_output_shapes
: *
T0
У
*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_3Exit,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_3*(
_output_shapes
:€€€€€€€€€»*
T0
У
*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_4Exit,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_4*(
_output_shapes
:€€€€€€€€€»*
T0
В
@Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3)Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_2*
_output_shapes
: *<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray
Ї
:Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/startConst*
value	B : *
dtype0*
_output_shapes
: *<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray
Ї
:Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray
Џ
4Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/rangeRange:Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/start@Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArraySizeV3:Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/delta*#
_output_shapes
:€€€€€€€€€*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray
П
BArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3)Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray4Arguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_2*
dtype0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€»*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray*%
element_shape:€€€€€€€€€»
p
%Arguments/BiLSTM/BiLSTM/fw/fw/Const_4Const*
_output_shapes
:*
dtype0*
valueB:»
f
$Arguments/BiLSTM/BiLSTM/fw/fw/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
m
+Arguments/BiLSTM/BiLSTM/fw/fw/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :
m
+Arguments/BiLSTM/BiLSTM/fw/fw/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
 
%Arguments/BiLSTM/BiLSTM/fw/fw/range_1Range+Arguments/BiLSTM/BiLSTM/fw/fw/range_1/start$Arguments/BiLSTM/BiLSTM/fw/fw/Rank_1+Arguments/BiLSTM/BiLSTM/fw/fw/range_1/delta*
_output_shapes
:
А
/Arguments/BiLSTM/BiLSTM/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
m
+Arguments/BiLSTM/BiLSTM/fw/fw/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
е
&Arguments/BiLSTM/BiLSTM/fw/fw/concat_2ConcatV2/Arguments/BiLSTM/BiLSTM/fw/fw/concat_2/values_0%Arguments/BiLSTM/BiLSTM/fw/fw/range_1+Arguments/BiLSTM/BiLSTM/fw/fw/concat_2/axis*
_output_shapes
:*
N*
T0
в
)Arguments/BiLSTM/BiLSTM/fw/fw/transpose_1	TransposeBArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArrayGatherV3&Arguments/BiLSTM/BiLSTM/fw/fw/concat_2*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€»*
T0
 
*Arguments/BiLSTM/BiLSTM/bw/ReverseSequenceReverseSequenceArguments/concat_2arg_original_sequence_lengths*
seq_dim*
T0*

Tlen0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€P
d
"Arguments/BiLSTM/BiLSTM/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
k
)Arguments/BiLSTM/BiLSTM/bw/bw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
k
)Arguments/BiLSTM/BiLSTM/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
¬
#Arguments/BiLSTM/BiLSTM/bw/bw/rangeRange)Arguments/BiLSTM/BiLSTM/bw/bw/range/start"Arguments/BiLSTM/BiLSTM/bw/bw/Rank)Arguments/BiLSTM/BiLSTM/bw/bw/range/delta*
_output_shapes
:
~
-Arguments/BiLSTM/BiLSTM/bw/bw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
k
)Arguments/BiLSTM/BiLSTM/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
$Arguments/BiLSTM/BiLSTM/bw/bw/concatConcatV2-Arguments/BiLSTM/BiLSTM/bw/bw/concat/values_0#Arguments/BiLSTM/BiLSTM/bw/bw/range)Arguments/BiLSTM/BiLSTM/bw/bw/concat/axis*
T0*
N*
_output_shapes
:
≈
'Arguments/BiLSTM/BiLSTM/bw/bw/transpose	Transpose*Arguments/BiLSTM/BiLSTM/bw/ReverseSequence$Arguments/BiLSTM/BiLSTM/bw/bw/concat*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€P
Ж
-Arguments/BiLSTM/BiLSTM/bw/bw/sequence_lengthIdentityarg_original_sequence_lengths*#
_output_shapes
:€€€€€€€€€*
T0
z
#Arguments/BiLSTM/BiLSTM/bw/bw/ShapeShape'Arguments/BiLSTM/BiLSTM/bw/bw/transpose*
_output_shapes
:*
T0
{
1Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
}
3Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
}
3Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
√
+Arguments/BiLSTM/BiLSTM/bw/bw/strided_sliceStridedSlice#Arguments/BiLSTM/BiLSTM/bw/bw/Shape1Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice/stack3Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_13Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
Э
[Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
Ф
WArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims
ExpandDims+Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice[Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dim*
_output_shapes
:*
T0
Э
RArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstConst*
valueB:»*
_output_shapes
:*
dtype0
Ъ
XArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ф
SArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatConcatV2WArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDimsRArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstXArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axis*
N*
T0*
_output_shapes
:
Э
XArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
RArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zerosFillSArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatXArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/Const*
T0*(
_output_shapes
:€€€€€€€€€»
Я
]Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
YArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1
ExpandDims+Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice]Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:
Я
TArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_1Const*
valueB:»*
dtype0*
_output_shapes
:
Я
]Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
YArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2
ExpandDims+Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice]Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dim*
T0*
_output_shapes
:
Я
TArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2Const*
valueB:»*
dtype0*
_output_shapes
:
Ь
ZArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
UArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1ConcatV2YArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2TArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2ZArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axis*
T0*
_output_shapes
:*
N
Я
ZArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¬
TArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1FillUArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1ZArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/Const*
T0*(
_output_shapes
:€€€€€€€€€»
Я
]Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dimConst*
dtype0*
value	B : *
_output_shapes
: 
Ш
YArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3
ExpandDims+Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice]Arguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:
Я
TArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_3Const*
dtype0*
valueB:»*
_output_shapes
:
В
%Arguments/BiLSTM/BiLSTM/bw/bw/Shape_1Shape-Arguments/BiLSTM/BiLSTM/bw/bw/sequence_length*
T0*
_output_shapes
:
Ж
#Arguments/BiLSTM/BiLSTM/bw/bw/stackPack+Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice*
T0*
N*
_output_shapes
:
Э
#Arguments/BiLSTM/BiLSTM/bw/bw/EqualEqual%Arguments/BiLSTM/BiLSTM/bw/bw/Shape_1#Arguments/BiLSTM/BiLSTM/bw/bw/stack*
_output_shapes
:*
T0
m
#Arguments/BiLSTM/BiLSTM/bw/bw/ConstConst*
dtype0*
valueB: *
_output_shapes
:
К
!Arguments/BiLSTM/BiLSTM/bw/bw/AllAll#Arguments/BiLSTM/BiLSTM/bw/bw/Equal#Arguments/BiLSTM/BiLSTM/bw/bw/Const*
_output_shapes
: 
Є
*Arguments/BiLSTM/BiLSTM/bw/bw/Assert/ConstConst*
dtype0*^
valueUBS BMExpected shape for Tensor Arguments/BiLSTM/BiLSTM/bw/bw/sequence_length:0 is *
_output_shapes
: 
}
,Arguments/BiLSTM/BiLSTM/bw/bw/Assert/Const_1Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
ј
2Arguments/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_0Const*
dtype0*^
valueUBS BMExpected shape for Tensor Arguments/BiLSTM/BiLSTM/bw/bw/sequence_length:0 is *
_output_shapes
: 
Г
2Arguments/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_2Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
Ы
+Arguments/BiLSTM/BiLSTM/bw/bw/Assert/AssertAssert!Arguments/BiLSTM/BiLSTM/bw/bw/All2Arguments/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_0#Arguments/BiLSTM/BiLSTM/bw/bw/stack2Arguments/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_2%Arguments/BiLSTM/BiLSTM/bw/bw/Shape_1*
T
2
ј
)Arguments/BiLSTM/BiLSTM/bw/bw/CheckSeqLenIdentity-Arguments/BiLSTM/BiLSTM/bw/bw/sequence_length,^Arguments/BiLSTM/BiLSTM/bw/bw/Assert/Assert*#
_output_shapes
:€€€€€€€€€*
T0
|
%Arguments/BiLSTM/BiLSTM/bw/bw/Shape_2Shape'Arguments/BiLSTM/BiLSTM/bw/bw/transpose*
_output_shapes
:*
T0
}
3Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0

5Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0

5Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ќ
-Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1StridedSlice%Arguments/BiLSTM/BiLSTM/bw/bw/Shape_23Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack5Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_15Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
|
%Arguments/BiLSTM/BiLSTM/bw/bw/Shape_3Shape'Arguments/BiLSTM/BiLSTM/bw/bw/transpose*
_output_shapes
:*
T0
}
3Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:

5Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0

5Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ќ
-Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2StridedSlice%Arguments/BiLSTM/BiLSTM/bw/bw/Shape_33Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack5Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_15Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
n
,Arguments/BiLSTM/BiLSTM/bw/bw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Є
(Arguments/BiLSTM/BiLSTM/bw/bw/ExpandDims
ExpandDims-Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_2,Arguments/BiLSTM/BiLSTM/bw/bw/ExpandDims/dim*
_output_shapes
:*
T0
p
%Arguments/BiLSTM/BiLSTM/bw/bw/Const_1Const*
_output_shapes
:*
dtype0*
valueB:»
m
+Arguments/BiLSTM/BiLSTM/bw/bw/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ё
&Arguments/BiLSTM/BiLSTM/bw/bw/concat_1ConcatV2(Arguments/BiLSTM/BiLSTM/bw/bw/ExpandDims%Arguments/BiLSTM/BiLSTM/bw/bw/Const_1+Arguments/BiLSTM/BiLSTM/bw/bw/concat_1/axis*
_output_shapes
:*
T0*
N
n
)Arguments/BiLSTM/BiLSTM/bw/bw/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
±
#Arguments/BiLSTM/BiLSTM/bw/bw/zerosFill&Arguments/BiLSTM/BiLSTM/bw/bw/concat_1)Arguments/BiLSTM/BiLSTM/bw/bw/zeros/Const*(
_output_shapes
:€€€€€€€€€»*
T0
o
%Arguments/BiLSTM/BiLSTM/bw/bw/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Ы
!Arguments/BiLSTM/BiLSTM/bw/bw/MinMin)Arguments/BiLSTM/BiLSTM/bw/bw/CheckSeqLen%Arguments/BiLSTM/BiLSTM/bw/bw/Const_2*
T0*
_output_shapes
: 
o
%Arguments/BiLSTM/BiLSTM/bw/bw/Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
Ы
!Arguments/BiLSTM/BiLSTM/bw/bw/MaxMax)Arguments/BiLSTM/BiLSTM/bw/bw/CheckSeqLen%Arguments/BiLSTM/BiLSTM/bw/bw/Const_3*
T0*
_output_shapes
: 
d
"Arguments/BiLSTM/BiLSTM/bw/bw/timeConst*
dtype0*
_output_shapes
: *
value	B : 
¶
)Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayTensorArrayV3-Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1*
dtype0*
_output_shapes

:: *
identical_element_shapes(*I
tensor_array_name42Arguments/BiLSTM/BiLSTM/bw/bw/dynamic_rnn/output_0*%
element_shape:€€€€€€€€€»
¶
+Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray_1TensorArrayV3-Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1*
dtype0*
_output_shapes

:: *H
tensor_array_name31Arguments/BiLSTM/BiLSTM/bw/bw/dynamic_rnn/input_0*
identical_element_shapes(*$
element_shape:€€€€€€€€€P
Н
6Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/ShapeShape'Arguments/BiLSTM/BiLSTM/bw/bw/transpose*
_output_shapes
:*
T0
О
DArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Р
FArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Р
FArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ґ
>Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice6Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/ShapeDArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stackFArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_1FArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
~
<Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
~
<Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
†
6Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/rangeRange<Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/start>Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice<Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:€€€€€€€€€
К
XArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3+Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray_16Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range'Arguments/BiLSTM/BiLSTM/bw/bw/transpose-Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray_1:1*
T0*
_output_shapes
: *:
_class0
.,loc:@Arguments/BiLSTM/BiLSTM/bw/bw/transpose
i
'Arguments/BiLSTM/BiLSTM/bw/bw/Maximum/xConst*
value	B :*
_output_shapes
: *
dtype0
Э
%Arguments/BiLSTM/BiLSTM/bw/bw/MaximumMaximum'Arguments/BiLSTM/BiLSTM/bw/bw/Maximum/x!Arguments/BiLSTM/BiLSTM/bw/bw/Max*
T0*
_output_shapes
: 
І
%Arguments/BiLSTM/BiLSTM/bw/bw/MinimumMinimum-Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1%Arguments/BiLSTM/BiLSTM/bw/bw/Maximum*
T0*
_output_shapes
: 
w
5Arguments/BiLSTM/BiLSTM/bw/bw/while/iteration_counterConst*
value	B : *
_output_shapes
: *
dtype0
и
)Arguments/BiLSTM/BiLSTM/bw/bw/while/EnterEnter5Arguments/BiLSTM/BiLSTM/bw/bw/while/iteration_counter*
T0*A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: *
parallel_iterations 
„
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_1Enter"Arguments/BiLSTM/BiLSTM/bw/bw/time*
T0*A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: *
parallel_iterations 
а
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_2Enter+Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray:1*
T0*A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: *
parallel_iterations 
Щ
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_3EnterRArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*(
_output_shapes
:€€€€€€€€€»*
T0
Ы
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_4EnterTArguments/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*(
_output_shapes
:€€€€€€€€€»*
T0
Љ
)Arguments/BiLSTM/BiLSTM/bw/bw/while/MergeMerge)Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter1Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration*
N*
_output_shapes
: : *
T0
¬
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_1Merge+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_13Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_1*
_output_shapes
: : *
T0*
N
¬
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_2Merge+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_23Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_2*
_output_shapes
: : *
T0*
N
‘
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_3Merge+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_33Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_3**
_output_shapes
:€€€€€€€€€»: *
T0*
N
‘
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_4Merge+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_43Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_4**
_output_shapes
:€€€€€€€€€»: *
T0*
N
ђ
(Arguments/BiLSTM/BiLSTM/bw/bw/while/LessLess)Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge.Arguments/BiLSTM/BiLSTM/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
ш
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Less/EnterEnter-Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1*
is_constant(*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: *
T0
≤
*Arguments/BiLSTM/BiLSTM/bw/bw/while/Less_1Less+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_10Arguments/BiLSTM/BiLSTM/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
т
0Arguments/BiLSTM/BiLSTM/bw/bw/while/Less_1/EnterEnter%Arguments/BiLSTM/BiLSTM/bw/bw/Minimum*
T0*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: *
is_constant(
™
.Arguments/BiLSTM/BiLSTM/bw/bw/while/LogicalAnd
LogicalAnd(Arguments/BiLSTM/BiLSTM/bw/bw/while/Less*Arguments/BiLSTM/BiLSTM/bw/bw/while/Less_1*
_output_shapes
: 
А
,Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCondLoopCond.Arguments/BiLSTM/BiLSTM/bw/bw/while/LogicalAnd*
_output_shapes
: 
о
*Arguments/BiLSTM/BiLSTM/bw/bw/while/SwitchSwitch)Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge,Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCond*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge*
_output_shapes
: : *
T0
ф
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_1Switch+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_1,Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_1*
_output_shapes
: : 
ф
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_2Switch+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_2,Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_2*
_output_shapes
: : 
Ш
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_3Switch+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_3,Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_3*<
_output_shapes*
(:€€€€€€€€€»:€€€€€€€€€»
Ш
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_4Switch+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_4,Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*>
_class4
20loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_4*<
_output_shapes*
(:€€€€€€€€€»:€€€€€€€€€»
З
,Arguments/BiLSTM/BiLSTM/bw/bw/while/IdentityIdentity,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch:1*
_output_shapes
: *
T0
Л
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_1Identity.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_1:1*
_output_shapes
: *
T0
Л
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_2Identity.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_2:1*
_output_shapes
: *
T0
Э
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_3Identity.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_3:1*(
_output_shapes
:€€€€€€€€€»*
T0
Э
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_4Identity.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_4:1*(
_output_shapes
:€€€€€€€€€»*
T0
Ъ
)Arguments/BiLSTM/BiLSTM/bw/bw/while/add/yConst-^Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
®
'Arguments/BiLSTM/BiLSTM/bw/bw/while/addAdd,Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity)Arguments/BiLSTM/BiLSTM/bw/bw/while/add/y*
_output_shapes
: *
T0
ђ
5Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3TensorArrayReadV3;Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_1=Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:€€€€€€€€€P
З
;Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/EnterEnter+Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray_1*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
is_constant(*
_output_shapes
:*
T0
≤
=Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1EnterXArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
T0*
_output_shapes
: *
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context
÷
0Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqualGreaterEqual.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_16Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Enter*
T0*#
_output_shapes
:€€€€€€€€€
Й
6Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/EnterEnter)Arguments/BiLSTM/BiLSTM/bw/bw/CheckSeqLen*
is_constant(*
T0*#
_output_shapes
:€€€€€€€€€*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context
ѕ
EArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"     *7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
Ѕ
CArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *¶Шљ*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
Ѕ
CArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *¶Ш=*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
Ц
MArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformEArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
T0* 
_output_shapes
:
Ш†*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
Ѓ
CArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/subSubCArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/maxCArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
¬
CArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/mulMulMArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformCArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
Ш†*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
і
?Arguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniformAddCArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/mulCArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
Ш†*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
±
$Arguments/BiLSTM/bw/lstm_cell/kernel
VariableV2*
shape:
Ш†* 
_output_shapes
:
Ш†*
dtype0*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel
А
+Arguments/BiLSTM/bw/lstm_cell/kernel/AssignAssign$Arguments/BiLSTM/bw/lstm_cell/kernel?Arguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
Ш†*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel*
T0
Ж
)Arguments/BiLSTM/bw/lstm_cell/kernel/readIdentity$Arguments/BiLSTM/bw/lstm_cell/kernel* 
_output_shapes
:
Ш†*
T0
Ї
4Arguments/BiLSTM/bw/lstm_cell/bias/Initializer/ConstConst*
valueB†*    *
_output_shapes	
:†*5
_class+
)'loc:@Arguments/BiLSTM/bw/lstm_cell/bias*
dtype0
£
"Arguments/BiLSTM/bw/lstm_cell/bias
VariableV2*
shape:†*
_output_shapes	
:†*5
_class+
)'loc:@Arguments/BiLSTM/bw/lstm_cell/bias*
dtype0
к
)Arguments/BiLSTM/bw/lstm_cell/bias/AssignAssign"Arguments/BiLSTM/bw/lstm_cell/bias4Arguments/BiLSTM/bw/lstm_cell/bias/Initializer/Const*
_output_shapes	
:†*
T0*5
_class+
)'loc:@Arguments/BiLSTM/bw/lstm_cell/bias
}
'Arguments/BiLSTM/bw/lstm_cell/bias/readIdentity"Arguments/BiLSTM/bw/lstm_cell/bias*
_output_shapes	
:†*
T0
±
3Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zerosConst-^Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity*
valueB»*    *
_output_shapes	
:»*
dtype0
ћ
;Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCellLSTMBlockCell5Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_3.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_4AArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter3Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zeros3Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zeros3Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zerosCArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1*Ґ
_output_shapesП
М:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»:€€€€€€€€€»*
T0*
	cell_clip%  Ањ
С
AArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/EnterEnter)Arguments/BiLSTM/bw/lstm_cell/kernel/read* 
_output_shapes
:
Ш†*A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
parallel_iterations *
is_constant(*
T0
М
CArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1Enter'Arguments/BiLSTM/bw/lstm_cell/bias/read*
_output_shapes	
:†*A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
parallel_iterations *
is_constant(*
T0
Џ
*Arguments/BiLSTM/BiLSTM/bw/bw/while/SelectSelect0Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual0Arguments/BiLSTM/BiLSTM/bw/bw/while/Select/Enter=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:6*(
_output_shapes
:€€€€€€€€€»*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
“
0Arguments/BiLSTM/BiLSTM/bw/bw/while/Select/EnterEnter#Arguments/BiLSTM/BiLSTM/bw/bw/zeros*(
_output_shapes
:€€€€€€€€€»*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*
parallel_iterations *
T0*
is_constant(
Џ
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Select_1Select0Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_3=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:1*(
_output_shapes
:€€€€€€€€€»*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
Џ
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Select_2Select0Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_4=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:6*(
_output_shapes
:€€€€€€€€€»*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
©
GArguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3MArguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_1*Arguments/BiLSTM/BiLSTM/bw/bw/while/Select.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_2*
_output_shapes
: *N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
з
MArguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter)Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray*
_output_shapes
:*
parallel_iterations *A

frame_name31Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context*N
_classD
B@loc:@Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0*
is_constant(
Ь
+Arguments/BiLSTM/BiLSTM/bw/bw/while/add_1/yConst-^Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ѓ
)Arguments/BiLSTM/BiLSTM/bw/bw/while/add_1Add.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_1+Arguments/BiLSTM/BiLSTM/bw/bw/while/add_1/y*
_output_shapes
: *
T0
М
1Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIterationNextIteration'Arguments/BiLSTM/BiLSTM/bw/bw/while/add*
_output_shapes
: *
T0
Р
3Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_1NextIteration)Arguments/BiLSTM/BiLSTM/bw/bw/while/add_1*
_output_shapes
: *
T0
Ѓ
3Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_2NextIterationGArguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
•
3Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_3NextIteration,Arguments/BiLSTM/BiLSTM/bw/bw/while/Select_1*(
_output_shapes
:€€€€€€€€€»*
T0
•
3Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_4NextIteration,Arguments/BiLSTM/BiLSTM/bw/bw/while/Select_2*(
_output_shapes
:€€€€€€€€€»*
T0
}
(Arguments/BiLSTM/BiLSTM/bw/bw/while/ExitExit*Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch*
_output_shapes
: *
T0
Б
*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_1Exit,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_1*
_output_shapes
: *
T0
Б
*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_2Exit,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_2*
_output_shapes
: *
T0
У
*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_3Exit,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_3*(
_output_shapes
:€€€€€€€€€»*
T0
У
*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_4Exit,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_4*(
_output_shapes
:€€€€€€€€€»*
T0
В
@Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3)Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_2*
_output_shapes
: *<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray
Ї
:Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/startConst*
_output_shapes
: *
value	B : *
dtype0*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray
Ї
:Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray
Џ
4Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/rangeRange:Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/start@Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArraySizeV3:Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/delta*#
_output_shapes
:€€€€€€€€€*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray
П
BArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3)Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray4Arguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_2*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€»*%
element_shape:€€€€€€€€€»*
dtype0*<
_class2
0.loc:@Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray
p
%Arguments/BiLSTM/BiLSTM/bw/bw/Const_4Const*
valueB:»*
_output_shapes
:*
dtype0
f
$Arguments/BiLSTM/BiLSTM/bw/bw/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
m
+Arguments/BiLSTM/BiLSTM/bw/bw/range_1/startConst*
value	B :*
_output_shapes
: *
dtype0
m
+Arguments/BiLSTM/BiLSTM/bw/bw/range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
 
%Arguments/BiLSTM/BiLSTM/bw/bw/range_1Range+Arguments/BiLSTM/BiLSTM/bw/bw/range_1/start$Arguments/BiLSTM/BiLSTM/bw/bw/Rank_1+Arguments/BiLSTM/BiLSTM/bw/bw/range_1/delta*
_output_shapes
:
А
/Arguments/BiLSTM/BiLSTM/bw/bw/concat_2/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
m
+Arguments/BiLSTM/BiLSTM/bw/bw/concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
е
&Arguments/BiLSTM/BiLSTM/bw/bw/concat_2ConcatV2/Arguments/BiLSTM/BiLSTM/bw/bw/concat_2/values_0%Arguments/BiLSTM/BiLSTM/bw/bw/range_1+Arguments/BiLSTM/BiLSTM/bw/bw/concat_2/axis*
T0*
_output_shapes
:*
N
в
)Arguments/BiLSTM/BiLSTM/bw/bw/transpose_1	TransposeBArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArrayGatherV3&Arguments/BiLSTM/BiLSTM/bw/bw/concat_2*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€»*
T0
Ў
 Arguments/BiLSTM/ReverseSequenceReverseSequence)Arguments/BiLSTM/BiLSTM/bw/bw/transpose_1arg_original_sequence_lengths*

Tlen0*
T0*
seq_dim*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€»
Z
Arguments/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
Ђ
Arguments/ExpandDims
ExpandDims)Arguments/BiLSTM/BiLSTM/fw/fw/transpose_1Arguments/ExpandDims/dim*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€»
\
Arguments/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
¶
Arguments/ExpandDims_1
ExpandDims Arguments/BiLSTM/ReverseSequenceArguments/ExpandDims_1/dim*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€»
b
Arguments/concat_3/axisConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
≤
Arguments/concat_3ConcatV2Arguments/ExpandDimsArguments/ExpandDims_1Arguments/concat_3/axis*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€»*
N
k
 Arguments/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
М
Arguments/MeanMeanArguments/concat_3 Arguments/Mean/reduction_indices*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€»
Ѓ
4Arguments/proj_1/W_1/Initializer/random_normal/shapeConst*
dtype0*'
_class
loc:@Arguments/proj_1/W_1*
_output_shapes
:*
valueB"»      
°
3Arguments/proj_1/W_1/Initializer/random_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0*'
_class
loc:@Arguments/proj_1/W_1
£
5Arguments/proj_1/W_1/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  А?*
dtype0*'
_class
loc:@Arguments/proj_1/W_1
с
CArguments/proj_1/W_1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal4Arguments/proj_1/W_1/Initializer/random_normal/shape*
_output_shapes
:	»*
T0*
dtype0*'
_class
loc:@Arguments/proj_1/W_1
И
2Arguments/proj_1/W_1/Initializer/random_normal/mulMulCArguments/proj_1/W_1/Initializer/random_normal/RandomStandardNormal5Arguments/proj_1/W_1/Initializer/random_normal/stddev*
_output_shapes
:	»*
T0*'
_class
loc:@Arguments/proj_1/W_1
с
.Arguments/proj_1/W_1/Initializer/random_normalAdd2Arguments/proj_1/W_1/Initializer/random_normal/mul3Arguments/proj_1/W_1/Initializer/random_normal/mean*
_output_shapes
:	»*
T0*'
_class
loc:@Arguments/proj_1/W_1
ґ
#Arguments/proj_1/W_1/Initializer/QrQr.Arguments/proj_1/W_1/Initializer/random_normal*)
_output_shapes
:	»:*
T0*'
_class
loc:@Arguments/proj_1/W_1
™
)Arguments/proj_1/W_1/Initializer/DiagPartDiagPart%Arguments/proj_1/W_1/Initializer/Qr:1*
_output_shapes
:*
T0*'
_class
loc:@Arguments/proj_1/W_1
¶
%Arguments/proj_1/W_1/Initializer/SignSign)Arguments/proj_1/W_1/Initializer/DiagPart*
_output_shapes
:*
T0*'
_class
loc:@Arguments/proj_1/W_1
 
$Arguments/proj_1/W_1/Initializer/mulMul#Arguments/proj_1/W_1/Initializer/Qr%Arguments/proj_1/W_1/Initializer/Sign*
_output_shapes
:	»*
T0*'
_class
loc:@Arguments/proj_1/W_1
®
.Arguments/proj_1/W_1/Initializer/Reshape/shapeConst*
valueB"»      *'
_class
loc:@Arguments/proj_1/W_1*
dtype0*
_output_shapes
:
№
(Arguments/proj_1/W_1/Initializer/ReshapeReshape$Arguments/proj_1/W_1/Initializer/mul.Arguments/proj_1/W_1/Initializer/Reshape/shape*'
_class
loc:@Arguments/proj_1/W_1*
T0*
_output_shapes
:	»
Ц
(Arguments/proj_1/W_1/Initializer/mul_1/xConst*
valueB
 *ЪЩ?*'
_class
loc:@Arguments/proj_1/W_1*
dtype0*
_output_shapes
: 
‘
&Arguments/proj_1/W_1/Initializer/mul_1Mul(Arguments/proj_1/W_1/Initializer/mul_1/x(Arguments/proj_1/W_1/Initializer/Reshape*'
_class
loc:@Arguments/proj_1/W_1*
T0*
_output_shapes
:	»
П
Arguments/proj_1/W_1
VariableV2*'
_class
loc:@Arguments/proj_1/W_1*
dtype0*
_output_shapes
:	»*
shape:	»
ґ
Arguments/proj_1/W_1/AssignAssignArguments/proj_1/W_1&Arguments/proj_1/W_1/Initializer/mul_1*
_output_shapes
:	»*
T0*'
_class
loc:@Arguments/proj_1/W_1
О
Arguments/proj_1/W_1/readIdentityArguments/proj_1/W_1*'
_class
loc:@Arguments/proj_1/W_1*
T0*
_output_shapes
:	»
Ь
&Arguments/proj_1/b_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@Arguments/proj_1/b_1*
dtype0*
_output_shapes
:
Е
Arguments/proj_1/b_1
VariableV2*
_output_shapes
:*
shape:*'
_class
loc:@Arguments/proj_1/b_1*
dtype0
±
Arguments/proj_1/b_1/AssignAssignArguments/proj_1/b_1&Arguments/proj_1/b_1/Initializer/zeros*
_output_shapes
:*
T0*'
_class
loc:@Arguments/proj_1/b_1
Й
Arguments/proj_1/b_1/readIdentityArguments/proj_1/b_1*
_output_shapes
:*
T0*'
_class
loc:@Arguments/proj_1/b_1
o
Arguments/proj_1/Reshape/shapeConst*
_output_shapes
:*
valueB"€€€€»   *
dtype0
Ж
Arguments/proj_1/ReshapeReshapeArguments/MeanArguments/proj_1/Reshape/shape*(
_output_shapes
:€€€€€€€€€»*
T0
И
Arguments/proj_1/MatMulMatMulArguments/proj_1/ReshapeArguments/proj_1/W_1/read*'
_output_shapes
:€€€€€€€€€*
T0
Б
Arguments/proj_1/addAddArguments/proj_1/MatMulArguments/proj_1/b_1/read*'
_output_shapes
:€€€€€€€€€*
T0
u
 Arguments/proj_1/Reshape_1/shapeConst*
_output_shapes
:*!
valueB"€€€€Я      *
dtype0
Ф
Arguments/proj_1/Reshape_1ReshapeArguments/proj_1/add Arguments/proj_1/Reshape_1/shape*,
_output_shapes
:€€€€€€€€€Я*
T0
Ј
9Arguments/CRF/transition/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *+
_class!
loc:@Arguments/CRF/transition*
dtype0
©
7Arguments/CRF/transition/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *.щдЊ*+
_class!
loc:@Arguments/CRF/transition*
dtype0
©
7Arguments/CRF/transition/Initializer/random_uniform/maxConst*+
_class!
loc:@Arguments/CRF/transition*
dtype0*
_output_shapes
: *
valueB
 *.щд>
р
AArguments/CRF/transition/Initializer/random_uniform/RandomUniformRandomUniform9Arguments/CRF/transition/Initializer/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*+
_class!
loc:@Arguments/CRF/transition
ю
7Arguments/CRF/transition/Initializer/random_uniform/subSub7Arguments/CRF/transition/Initializer/random_uniform/max7Arguments/CRF/transition/Initializer/random_uniform/min*
T0*
_output_shapes
: *+
_class!
loc:@Arguments/CRF/transition
Р
7Arguments/CRF/transition/Initializer/random_uniform/mulMulAArguments/CRF/transition/Initializer/random_uniform/RandomUniform7Arguments/CRF/transition/Initializer/random_uniform/sub*
T0*
_output_shapes

:*+
_class!
loc:@Arguments/CRF/transition
В
3Arguments/CRF/transition/Initializer/random_uniformAdd7Arguments/CRF/transition/Initializer/random_uniform/mul7Arguments/CRF/transition/Initializer/random_uniform/min*
T0*
_output_shapes

:*+
_class!
loc:@Arguments/CRF/transition
Х
Arguments/CRF/transition
VariableV2*
dtype0*
shape
:*
_output_shapes

:*+
_class!
loc:@Arguments/CRF/transition
ќ
Arguments/CRF/transition/AssignAssignArguments/CRF/transition3Arguments/CRF/transition/Initializer/random_uniform*
T0*
_output_shapes

:*+
_class!
loc:@Arguments/CRF/transition
Щ
Arguments/CRF/transition/readIdentityArguments/CRF/transition*
T0*
_output_shapes

:*+
_class!
loc:@Arguments/CRF/transition
X
Arguments/CRF/Equal/xConst*
dtype0*
_output_shapes
: *
value
B :Я
W
Arguments/CRF/Equal/yConst*
dtype0*
_output_shapes
: *
value	B :
k
Arguments/CRF/EqualEqualArguments/CRF/Equal/xArguments/CRF/Equal/y*
T0*
_output_shapes
: 
^
Arguments/CRF/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Р
Arguments/CRF/ExpandDims
ExpandDimsArguments/CRF/transition/readArguments/CRF/ExpandDims/dim*"
_output_shapes
:*
T0
n
Arguments/CRF/Slice/beginConst*
dtype0*
_output_shapes
:*!
valueB"            
m
Arguments/CRF/Slice/sizeConst*
dtype0*
_output_shapes
:*!
valueB"€€€€   €€€€
∞
Arguments/CRF/SliceSliceArguments/proj_1/Reshape_1Arguments/CRF/Slice/beginArguments/CRF/Slice/size*
Index0*+
_output_shapes
:€€€€€€€€€*
T0
~
Arguments/CRF/SqueezeSqueezeArguments/CRF/Slice*
squeeze_dims
*'
_output_shapes
:€€€€€€€€€*
T0
p
Arguments/CRF/Slice_1/beginConst*
dtype0*
_output_shapes
:*!
valueB"           
o
Arguments/CRF/Slice_1/sizeConst*
dtype0*
_output_shapes
:*!
valueB"€€€€€€€€€€€€
Ј
Arguments/CRF/Slice_1SliceArguments/proj_1/Reshape_1Arguments/CRF/Slice_1/beginArguments/CRF/Slice_1/size*
Index0*,
_output_shapes
:€€€€€€€€€Ю*
T0
U
Arguments/CRF/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
U
Arguments/CRF/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
z
Arguments/CRF/subSubarg_original_sequence_lengthsArguments/CRF/sub/y*#
_output_shapes
:€€€€€€€€€*
T0
v
Arguments/CRF/MaximumMaximumArguments/CRF/ConstArguments/CRF/sub*
T0*#
_output_shapes
:€€€€€€€€€
X
Arguments/CRF/rnn/RankConst*
dtype0*
_output_shapes
: *
value	B :
_
Arguments/CRF/rnn/range/startConst*
dtype0*
_output_shapes
: *
value	B :
_
Arguments/CRF/rnn/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Т
Arguments/CRF/rnn/rangeRangeArguments/CRF/rnn/range/startArguments/CRF/rnn/RankArguments/CRF/rnn/range/delta*
_output_shapes
:
r
!Arguments/CRF/rnn/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
_
Arguments/CRF/rnn/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
≠
Arguments/CRF/rnn/concatConcatV2!Arguments/CRF/rnn/concat/values_0Arguments/CRF/rnn/rangeArguments/CRF/rnn/concat/axis*
N*
T0*
_output_shapes
:
Р
Arguments/CRF/rnn/transpose	TransposeArguments/CRF/Slice_1Arguments/CRF/rnn/concat*
T0*,
_output_shapes
:Ю€€€€€€€€€
r
!Arguments/CRF/rnn/sequence_lengthIdentityArguments/CRF/Maximum*
T0*#
_output_shapes
:€€€€€€€€€
b
Arguments/CRF/rnn/ShapeShapeArguments/CRF/rnn/transpose*
T0*
_output_shapes
:
o
%Arguments/CRF/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
q
'Arguments/CRF/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
q
'Arguments/CRF/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
З
Arguments/CRF/rnn/strided_sliceStridedSliceArguments/CRF/rnn/Shape%Arguments/CRF/rnn/strided_slice/stack'Arguments/CRF/rnn/strided_slice/stack_1'Arguments/CRF/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
j
Arguments/CRF/rnn/Shape_1Shape!Arguments/CRF/rnn/sequence_length*
T0*
_output_shapes
:
n
Arguments/CRF/rnn/stackPackArguments/CRF/rnn/strided_slice*
T0*
_output_shapes
:*
N
y
Arguments/CRF/rnn/EqualEqualArguments/CRF/rnn/Shape_1Arguments/CRF/rnn/stack*
T0*
_output_shapes
:
a
Arguments/CRF/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
f
Arguments/CRF/rnn/AllAllArguments/CRF/rnn/EqualArguments/CRF/rnn/Const*
_output_shapes
: 
†
Arguments/CRF/rnn/Assert/ConstConst*
dtype0*
_output_shapes
: *R
valueIBG BAExpected shape for Tensor Arguments/CRF/rnn/sequence_length:0 is 
q
 Arguments/CRF/rnn/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
®
&Arguments/CRF/rnn/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *R
valueIBG BAExpected shape for Tensor Arguments/CRF/rnn/sequence_length:0 is 
w
&Arguments/CRF/rnn/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
”
Arguments/CRF/rnn/Assert/AssertAssertArguments/CRF/rnn/All&Arguments/CRF/rnn/Assert/Assert/data_0Arguments/CRF/rnn/stack&Arguments/CRF/rnn/Assert/Assert/data_2Arguments/CRF/rnn/Shape_1*
T
2
Ь
Arguments/CRF/rnn/CheckSeqLenIdentity!Arguments/CRF/rnn/sequence_length ^Arguments/CRF/rnn/Assert/Assert*#
_output_shapes
:€€€€€€€€€*
T0
d
Arguments/CRF/rnn/Shape_2ShapeArguments/CRF/rnn/transpose*
_output_shapes
:*
T0
q
'Arguments/CRF/rnn/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0
s
)Arguments/CRF/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
s
)Arguments/CRF/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
С
!Arguments/CRF/rnn/strided_slice_1StridedSliceArguments/CRF/rnn/Shape_2'Arguments/CRF/rnn/strided_slice_1/stack)Arguments/CRF/rnn/strided_slice_1/stack_1)Arguments/CRF/rnn/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0
d
Arguments/CRF/rnn/Shape_3ShapeArguments/CRF/rnn/transpose*
_output_shapes
:*
T0
q
'Arguments/CRF/rnn/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0
s
)Arguments/CRF/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
s
)Arguments/CRF/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
С
!Arguments/CRF/rnn/strided_slice_2StridedSliceArguments/CRF/rnn/Shape_3'Arguments/CRF/rnn/strided_slice_2/stack)Arguments/CRF/rnn/strided_slice_2/stack_1)Arguments/CRF/rnn/strided_slice_2/stack_2*
_output_shapes
: *
T0*
shrink_axis_mask*
Index0
b
 Arguments/CRF/rnn/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
Ф
Arguments/CRF/rnn/ExpandDims
ExpandDims!Arguments/CRF/rnn/strided_slice_2 Arguments/CRF/rnn/ExpandDims/dim*
_output_shapes
:*
T0
c
Arguments/CRF/rnn/Const_1Const*
valueB:*
_output_shapes
:*
dtype0
a
Arguments/CRF/rnn/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ѓ
Arguments/CRF/rnn/concat_1ConcatV2Arguments/CRF/rnn/ExpandDimsArguments/CRF/rnn/Const_1Arguments/CRF/rnn/concat_1/axis*
N*
_output_shapes
:*
T0
_
Arguments/CRF/rnn/zeros/ConstConst*
value	B : *
_output_shapes
: *
dtype0
М
Arguments/CRF/rnn/zerosFillArguments/CRF/rnn/concat_1Arguments/CRF/rnn/zeros/Const*'
_output_shapes
:€€€€€€€€€*
T0
c
Arguments/CRF/rnn/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
w
Arguments/CRF/rnn/MinMinArguments/CRF/rnn/CheckSeqLenArguments/CRF/rnn/Const_2*
_output_shapes
: *
T0
c
Arguments/CRF/rnn/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
w
Arguments/CRF/rnn/MaxMaxArguments/CRF/rnn/CheckSeqLenArguments/CRF/rnn/Const_3*
T0*
_output_shapes
: 
X
Arguments/CRF/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Б
Arguments/CRF/rnn/TensorArrayTensorArrayV3!Arguments/CRF/rnn/strided_slice_1*$
element_shape:€€€€€€€€€*
dtype0*
identical_element_shapes(*=
tensor_array_name(&Arguments/CRF/rnn/dynamic_rnn/output_0*
_output_shapes

:: 
В
Arguments/CRF/rnn/TensorArray_1TensorArrayV3!Arguments/CRF/rnn/strided_slice_1*$
element_shape:€€€€€€€€€*
dtype0*
identical_element_shapes(*<
tensor_array_name'%Arguments/CRF/rnn/dynamic_rnn/input_0*
_output_shapes

:: 
u
*Arguments/CRF/rnn/TensorArrayUnstack/ShapeShapeArguments/CRF/rnn/transpose*
T0*
_output_shapes
:
В
8Arguments/CRF/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Д
:Arguments/CRF/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Д
:Arguments/CRF/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ж
2Arguments/CRF/rnn/TensorArrayUnstack/strided_sliceStridedSlice*Arguments/CRF/rnn/TensorArrayUnstack/Shape8Arguments/CRF/rnn/TensorArrayUnstack/strided_slice/stack:Arguments/CRF/rnn/TensorArrayUnstack/strided_slice/stack_1:Arguments/CRF/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
r
0Arguments/CRF/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
r
0Arguments/CRF/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
р
*Arguments/CRF/rnn/TensorArrayUnstack/rangeRange0Arguments/CRF/rnn/TensorArrayUnstack/range/start2Arguments/CRF/rnn/TensorArrayUnstack/strided_slice0Arguments/CRF/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:€€€€€€€€€
¬
LArguments/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Arguments/CRF/rnn/TensorArray_1*Arguments/CRF/rnn/TensorArrayUnstack/rangeArguments/CRF/rnn/transpose!Arguments/CRF/rnn/TensorArray_1:1*
T0*
_output_shapes
: *.
_class$
" loc:@Arguments/CRF/rnn/transpose
]
Arguments/CRF/rnn/Maximum/xConst*
_output_shapes
: *
value	B :*
dtype0
y
Arguments/CRF/rnn/MaximumMaximumArguments/CRF/rnn/Maximum/xArguments/CRF/rnn/Max*
T0*
_output_shapes
: 
Г
Arguments/CRF/rnn/MinimumMinimum!Arguments/CRF/rnn/strided_slice_1Arguments/CRF/rnn/Maximum*
T0*
_output_shapes
: 
k
)Arguments/CRF/rnn/while/iteration_counterConst*
_output_shapes
: *
value	B : *
dtype0
ƒ
Arguments/CRF/rnn/while/EnterEnter)Arguments/CRF/rnn/while/iteration_counter*
parallel_iterations *
_output_shapes
: *5

frame_name'%Arguments/CRF/rnn/while/while_context*
T0
≥
Arguments/CRF/rnn/while/Enter_1EnterArguments/CRF/rnn/time*
parallel_iterations *
_output_shapes
: *5

frame_name'%Arguments/CRF/rnn/while/while_context*
T0
Љ
Arguments/CRF/rnn/while/Enter_2EnterArguments/CRF/rnn/TensorArray:1*
parallel_iterations *
_output_shapes
: *5

frame_name'%Arguments/CRF/rnn/while/while_context*
T0
√
Arguments/CRF/rnn/while/Enter_3EnterArguments/CRF/Squeeze*
parallel_iterations *'
_output_shapes
:€€€€€€€€€*5

frame_name'%Arguments/CRF/rnn/while/while_context*
T0
Ш
Arguments/CRF/rnn/while/MergeMergeArguments/CRF/rnn/while/Enter%Arguments/CRF/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
Ю
Arguments/CRF/rnn/while/Merge_1MergeArguments/CRF/rnn/while/Enter_1'Arguments/CRF/rnn/while/NextIteration_1*
N*
_output_shapes
: : *
T0
Ю
Arguments/CRF/rnn/while/Merge_2MergeArguments/CRF/rnn/while/Enter_2'Arguments/CRF/rnn/while/NextIteration_2*
T0*
_output_shapes
: : *
N
ѓ
Arguments/CRF/rnn/while/Merge_3MergeArguments/CRF/rnn/while/Enter_3'Arguments/CRF/rnn/while/NextIteration_3*
N*)
_output_shapes
:€€€€€€€€€: *
T0
И
Arguments/CRF/rnn/while/LessLessArguments/CRF/rnn/while/Merge"Arguments/CRF/rnn/while/Less/Enter*
_output_shapes
: *
T0
‘
"Arguments/CRF/rnn/while/Less/EnterEnter!Arguments/CRF/rnn/strided_slice_1*
is_constant(*
parallel_iterations *
_output_shapes
: *
T0*5

frame_name'%Arguments/CRF/rnn/while/while_context
О
Arguments/CRF/rnn/while/Less_1LessArguments/CRF/rnn/while/Merge_1$Arguments/CRF/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
ќ
$Arguments/CRF/rnn/while/Less_1/EnterEnterArguments/CRF/rnn/Minimum*5

frame_name'%Arguments/CRF/rnn/while/while_context*
_output_shapes
: *
T0*
is_constant(*
parallel_iterations 
Ж
"Arguments/CRF/rnn/while/LogicalAnd
LogicalAndArguments/CRF/rnn/while/LessArguments/CRF/rnn/while/Less_1*
_output_shapes
: 
h
 Arguments/CRF/rnn/while/LoopCondLoopCond"Arguments/CRF/rnn/while/LogicalAnd*
_output_shapes
: 
Њ
Arguments/CRF/rnn/while/SwitchSwitchArguments/CRF/rnn/while/Merge Arguments/CRF/rnn/while/LoopCond*0
_class&
$"loc:@Arguments/CRF/rnn/while/Merge*
T0*
_output_shapes
: : 
ƒ
 Arguments/CRF/rnn/while/Switch_1SwitchArguments/CRF/rnn/while/Merge_1 Arguments/CRF/rnn/while/LoopCond*2
_class(
&$loc:@Arguments/CRF/rnn/while/Merge_1*
T0*
_output_shapes
: : 
ƒ
 Arguments/CRF/rnn/while/Switch_2SwitchArguments/CRF/rnn/while/Merge_2 Arguments/CRF/rnn/while/LoopCond*2
_class(
&$loc:@Arguments/CRF/rnn/while/Merge_2*
T0*
_output_shapes
: : 
ж
 Arguments/CRF/rnn/while/Switch_3SwitchArguments/CRF/rnn/while/Merge_3 Arguments/CRF/rnn/while/LoopCond*2
_class(
&$loc:@Arguments/CRF/rnn/while/Merge_3*
T0*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€
o
 Arguments/CRF/rnn/while/IdentityIdentity Arguments/CRF/rnn/while/Switch:1*
T0*
_output_shapes
: 
s
"Arguments/CRF/rnn/while/Identity_1Identity"Arguments/CRF/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
s
"Arguments/CRF/rnn/while/Identity_2Identity"Arguments/CRF/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
Д
"Arguments/CRF/rnn/while/Identity_3Identity"Arguments/CRF/rnn/while/Switch_3:1*'
_output_shapes
:€€€€€€€€€*
T0
В
Arguments/CRF/rnn/while/add/yConst!^Arguments/CRF/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
Д
Arguments/CRF/rnn/while/addAdd Arguments/CRF/rnn/while/IdentityArguments/CRF/rnn/while/add/y*
_output_shapes
: *
T0
ь
)Arguments/CRF/rnn/while/TensorArrayReadV3TensorArrayReadV3/Arguments/CRF/rnn/while/TensorArrayReadV3/Enter"Arguments/CRF/rnn/while/Identity_11Arguments/CRF/rnn/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:€€€€€€€€€*
dtype0
г
/Arguments/CRF/rnn/while/TensorArrayReadV3/EnterEnterArguments/CRF/rnn/TensorArray_1*5

frame_name'%Arguments/CRF/rnn/while/while_context*
_output_shapes
:*
T0*
is_constant(*
parallel_iterations 
О
1Arguments/CRF/rnn/while/TensorArrayReadV3/Enter_1EnterLArguments/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*5

frame_name'%Arguments/CRF/rnn/while/while_context*
_output_shapes
: *
T0*
is_constant(*
parallel_iterations 
≤
$Arguments/CRF/rnn/while/GreaterEqualGreaterEqual"Arguments/CRF/rnn/while/Identity_1*Arguments/CRF/rnn/while/GreaterEqual/Enter*#
_output_shapes
:€€€€€€€€€*
T0
е
*Arguments/CRF/rnn/while/GreaterEqual/EnterEnterArguments/CRF/rnn/CheckSeqLen*5

frame_name'%Arguments/CRF/rnn/while/while_context*#
_output_shapes
:€€€€€€€€€*
T0*
is_constant(*
parallel_iterations 
Л
&Arguments/CRF/rnn/while/ExpandDims/dimConst!^Arguments/CRF/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
≤
"Arguments/CRF/rnn/while/ExpandDims
ExpandDims"Arguments/CRF/rnn/while/Identity_3&Arguments/CRF/rnn/while/ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*
T0
£
Arguments/CRF/rnn/while/add_1Add"Arguments/CRF/rnn/while/ExpandDims#Arguments/CRF/rnn/while/add_1/Enter*
T0*+
_output_shapes
:€€€€€€€€€
Ў
#Arguments/CRF/rnn/while/add_1/EnterEnterArguments/CRF/ExpandDims*
T0*5

frame_name'%Arguments/CRF/rnn/while/while_context*"
_output_shapes
:*
parallel_iterations *
is_constant(
Ъ
-Arguments/CRF/rnn/while/Max/reduction_indicesConst!^Arguments/CRF/rnn/while/Identity*
_output_shapes
:*
valueB:*
dtype0
Ґ
Arguments/CRF/rnn/while/MaxMaxArguments/CRF/rnn/while/add_1-Arguments/CRF/rnn/while/Max/reduction_indices*
T0*'
_output_shapes
:€€€€€€€€€
Ю
Arguments/CRF/rnn/while/add_2Add)Arguments/CRF/rnn/while/TensorArrayReadV3Arguments/CRF/rnn/while/Max*
T0*'
_output_shapes
:€€€€€€€€€
Н
(Arguments/CRF/rnn/while/ArgMax/dimensionConst!^Arguments/CRF/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
£
Arguments/CRF/rnn/while/ArgMaxArgMaxArguments/CRF/rnn/while/add_1(Arguments/CRF/rnn/while/ArgMax/dimension*
T0*'
_output_shapes
:€€€€€€€€€
Е
Arguments/CRF/rnn/while/CastCastArguments/CRF/rnn/while/ArgMax*'
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
х
Arguments/CRF/rnn/while/SelectSelect$Arguments/CRF/rnn/while/GreaterEqual$Arguments/CRF/rnn/while/Select/EnterArguments/CRF/rnn/while/Cast*'
_output_shapes
:€€€€€€€€€*
T0*/
_class%
#!loc:@Arguments/CRF/rnn/while/Cast
О
$Arguments/CRF/rnn/while/Select/EnterEnterArguments/CRF/rnn/zeros*
is_constant(*
T0*/
_class%
#!loc:@Arguments/CRF/rnn/while/Cast*
parallel_iterations *'
_output_shapes
:€€€€€€€€€*5

frame_name'%Arguments/CRF/rnn/while/while_context
ч
 Arguments/CRF/rnn/while/Select_1Select$Arguments/CRF/rnn/while/GreaterEqual"Arguments/CRF/rnn/while/Identity_3Arguments/CRF/rnn/while/add_2*'
_output_shapes
:€€€€€€€€€*
T0*0
_class&
$"loc:@Arguments/CRF/rnn/while/add_2
ќ
;Arguments/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3AArguments/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter"Arguments/CRF/rnn/while/Identity_1Arguments/CRF/rnn/while/Select"Arguments/CRF/rnn/while/Identity_2*
_output_shapes
: *
T0*/
_class%
#!loc:@Arguments/CRF/rnn/while/Cast
§
AArguments/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterArguments/CRF/rnn/TensorArray*
is_constant(*
T0*/
_class%
#!loc:@Arguments/CRF/rnn/while/Cast*
parallel_iterations *
_output_shapes
:*5

frame_name'%Arguments/CRF/rnn/while/while_context
Д
Arguments/CRF/rnn/while/add_3/yConst!^Arguments/CRF/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
К
Arguments/CRF/rnn/while/add_3Add"Arguments/CRF/rnn/while/Identity_1Arguments/CRF/rnn/while/add_3/y*
_output_shapes
: *
T0
t
%Arguments/CRF/rnn/while/NextIterationNextIterationArguments/CRF/rnn/while/add*
T0*
_output_shapes
: 
x
'Arguments/CRF/rnn/while/NextIteration_1NextIterationArguments/CRF/rnn/while/add_3*
T0*
_output_shapes
: 
Ц
'Arguments/CRF/rnn/while/NextIteration_2NextIteration;Arguments/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
М
'Arguments/CRF/rnn/while/NextIteration_3NextIteration Arguments/CRF/rnn/while/Select_1*
T0*'
_output_shapes
:€€€€€€€€€
e
Arguments/CRF/rnn/while/ExitExitArguments/CRF/rnn/while/Switch*
T0*
_output_shapes
: 
i
Arguments/CRF/rnn/while/Exit_1Exit Arguments/CRF/rnn/while/Switch_1*
T0*
_output_shapes
: 
i
Arguments/CRF/rnn/while/Exit_2Exit Arguments/CRF/rnn/while/Switch_2*
T0*
_output_shapes
: 
z
Arguments/CRF/rnn/while/Exit_3Exit Arguments/CRF/rnn/while/Switch_3*
T0*'
_output_shapes
:€€€€€€€€€
“
4Arguments/CRF/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3Arguments/CRF/rnn/TensorArrayArguments/CRF/rnn/while/Exit_2*
_output_shapes
: *0
_class&
$"loc:@Arguments/CRF/rnn/TensorArray
Ґ
.Arguments/CRF/rnn/TensorArrayStack/range/startConst*
_output_shapes
: *0
_class&
$"loc:@Arguments/CRF/rnn/TensorArray*
value	B : *
dtype0
Ґ
.Arguments/CRF/rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: *0
_class&
$"loc:@Arguments/CRF/rnn/TensorArray*
value	B :*
dtype0
Ю
(Arguments/CRF/rnn/TensorArrayStack/rangeRange.Arguments/CRF/rnn/TensorArrayStack/range/start4Arguments/CRF/rnn/TensorArrayStack/TensorArraySizeV3.Arguments/CRF/rnn/TensorArrayStack/range/delta*#
_output_shapes
:€€€€€€€€€*0
_class&
$"loc:@Arguments/CRF/rnn/TensorArray
…
6Arguments/CRF/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3Arguments/CRF/rnn/TensorArray(Arguments/CRF/rnn/TensorArrayStack/rangeArguments/CRF/rnn/while/Exit_2*,
_output_shapes
:Ю€€€€€€€€€*0
_class&
$"loc:@Arguments/CRF/rnn/TensorArray*
dtype0*$
element_shape:€€€€€€€€€
c
Arguments/CRF/rnn/Const_4Const*
_output_shapes
:*
valueB:*
dtype0
Z
Arguments/CRF/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
a
Arguments/CRF/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
a
Arguments/CRF/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ъ
Arguments/CRF/rnn/range_1RangeArguments/CRF/rnn/range_1/startArguments/CRF/rnn/Rank_1Arguments/CRF/rnn/range_1/delta*
_output_shapes
:
t
#Arguments/CRF/rnn/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
a
Arguments/CRF/rnn/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
µ
Arguments/CRF/rnn/concat_2ConcatV2#Arguments/CRF/rnn/concat_2/values_0Arguments/CRF/rnn/range_1Arguments/CRF/rnn/concat_2/axis*
T0*
N*
_output_shapes
:
µ
Arguments/CRF/rnn/transpose_1	Transpose6Arguments/CRF/rnn/TensorArrayStack/TensorArrayGatherV3Arguments/CRF/rnn/concat_2*,
_output_shapes
:€€€€€€€€€Ю*
T0
Є
Arguments/CRF/ReverseSequenceReverseSequenceArguments/CRF/rnn/transpose_1Arguments/CRF/Maximum*

Tlen0*
seq_dim*
T0*,
_output_shapes
:€€€€€€€€€Ю
`
Arguments/CRF/ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
М
Arguments/CRF/ArgMaxArgMaxArguments/CRF/rnn/while/Exit_3Arguments/CRF/ArgMax/dimension*#
_output_shapes
:€€€€€€€€€*
T0
m
Arguments/CRF/CastCastArguments/CRF/ArgMax*#
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
i
Arguments/CRF/ExpandDims_1/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
О
Arguments/CRF/ExpandDims_1
ExpandDimsArguments/CRF/CastArguments/CRF/ExpandDims_1/dim*
T0*'
_output_shapes
:€€€€€€€€€
Z
Arguments/CRF/rnn_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
a
Arguments/CRF/rnn_1/range/startConst*
value	B :*
_output_shapes
: *
dtype0
a
Arguments/CRF/rnn_1/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Ъ
Arguments/CRF/rnn_1/rangeRangeArguments/CRF/rnn_1/range/startArguments/CRF/rnn_1/RankArguments/CRF/rnn_1/range/delta*
_output_shapes
:
t
#Arguments/CRF/rnn_1/concat/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
a
Arguments/CRF/rnn_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
µ
Arguments/CRF/rnn_1/concatConcatV2#Arguments/CRF/rnn_1/concat/values_0Arguments/CRF/rnn_1/rangeArguments/CRF/rnn_1/concat/axis*
N*
T0*
_output_shapes
:
Ь
Arguments/CRF/rnn_1/transpose	TransposeArguments/CRF/ReverseSequenceArguments/CRF/rnn_1/concat*
T0*,
_output_shapes
:Ю€€€€€€€€€
t
#Arguments/CRF/rnn_1/sequence_lengthIdentityArguments/CRF/Maximum*
T0*#
_output_shapes
:€€€€€€€€€
f
Arguments/CRF/rnn_1/ShapeShapeArguments/CRF/rnn_1/transpose*
_output_shapes
:*
T0
q
'Arguments/CRF/rnn_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)Arguments/CRF/rnn_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)Arguments/CRF/rnn_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
С
!Arguments/CRF/rnn_1/strided_sliceStridedSliceArguments/CRF/rnn_1/Shape'Arguments/CRF/rnn_1/strided_slice/stack)Arguments/CRF/rnn_1/strided_slice/stack_1)Arguments/CRF/rnn_1/strided_slice/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
n
Arguments/CRF/rnn_1/Shape_1Shape#Arguments/CRF/rnn_1/sequence_length*
T0*
_output_shapes
:
r
Arguments/CRF/rnn_1/stackPack!Arguments/CRF/rnn_1/strided_slice*
_output_shapes
:*
N*
T0

Arguments/CRF/rnn_1/EqualEqualArguments/CRF/rnn_1/Shape_1Arguments/CRF/rnn_1/stack*
T0*
_output_shapes
:
c
Arguments/CRF/rnn_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
l
Arguments/CRF/rnn_1/AllAllArguments/CRF/rnn_1/EqualArguments/CRF/rnn_1/Const*
_output_shapes
: 
§
 Arguments/CRF/rnn_1/Assert/ConstConst*
_output_shapes
: *
dtype0*T
valueKBI BCExpected shape for Tensor Arguments/CRF/rnn_1/sequence_length:0 is 
s
"Arguments/CRF/rnn_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*!
valueB B but saw shape: 
ђ
(Arguments/CRF/rnn_1/Assert/Assert/data_0Const*T
valueKBI BCExpected shape for Tensor Arguments/CRF/rnn_1/sequence_length:0 is *
_output_shapes
: *
dtype0
y
(Arguments/CRF/rnn_1/Assert/Assert/data_2Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
я
!Arguments/CRF/rnn_1/Assert/AssertAssertArguments/CRF/rnn_1/All(Arguments/CRF/rnn_1/Assert/Assert/data_0Arguments/CRF/rnn_1/stack(Arguments/CRF/rnn_1/Assert/Assert/data_2Arguments/CRF/rnn_1/Shape_1*
T
2
Ґ
Arguments/CRF/rnn_1/CheckSeqLenIdentity#Arguments/CRF/rnn_1/sequence_length"^Arguments/CRF/rnn_1/Assert/Assert*
T0*#
_output_shapes
:€€€€€€€€€
h
Arguments/CRF/rnn_1/Shape_2ShapeArguments/CRF/rnn_1/transpose*
T0*
_output_shapes
:
s
)Arguments/CRF/rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
u
+Arguments/CRF/rnn_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+Arguments/CRF/rnn_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ы
#Arguments/CRF/rnn_1/strided_slice_1StridedSliceArguments/CRF/rnn_1/Shape_2)Arguments/CRF/rnn_1/strided_slice_1/stack+Arguments/CRF/rnn_1/strided_slice_1/stack_1+Arguments/CRF/rnn_1/strided_slice_1/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
h
Arguments/CRF/rnn_1/Shape_3ShapeArguments/CRF/rnn_1/transpose*
_output_shapes
:*
T0
s
)Arguments/CRF/rnn_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
u
+Arguments/CRF/rnn_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+Arguments/CRF/rnn_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ы
#Arguments/CRF/rnn_1/strided_slice_2StridedSliceArguments/CRF/rnn_1/Shape_3)Arguments/CRF/rnn_1/strided_slice_2/stack+Arguments/CRF/rnn_1/strided_slice_2/stack_1+Arguments/CRF/rnn_1/strided_slice_2/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
d
"Arguments/CRF/rnn_1/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ъ
Arguments/CRF/rnn_1/ExpandDims
ExpandDims#Arguments/CRF/rnn_1/strided_slice_2"Arguments/CRF/rnn_1/ExpandDims/dim*
T0*
_output_shapes
:
e
Arguments/CRF/rnn_1/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
c
!Arguments/CRF/rnn_1/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ґ
Arguments/CRF/rnn_1/concat_1ConcatV2Arguments/CRF/rnn_1/ExpandDimsArguments/CRF/rnn_1/Const_1!Arguments/CRF/rnn_1/concat_1/axis*
T0*
_output_shapes
:*
N
a
Arguments/CRF/rnn_1/zeros/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
Т
Arguments/CRF/rnn_1/zerosFillArguments/CRF/rnn_1/concat_1Arguments/CRF/rnn_1/zeros/Const*'
_output_shapes
:€€€€€€€€€*
T0
e
Arguments/CRF/rnn_1/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
}
Arguments/CRF/rnn_1/MinMinArguments/CRF/rnn_1/CheckSeqLenArguments/CRF/rnn_1/Const_2*
_output_shapes
: *
T0
e
Arguments/CRF/rnn_1/Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
}
Arguments/CRF/rnn_1/MaxMaxArguments/CRF/rnn_1/CheckSeqLenArguments/CRF/rnn_1/Const_3*
_output_shapes
: *
T0
Z
Arguments/CRF/rnn_1/timeConst*
dtype0*
_output_shapes
: *
value	B : 
З
Arguments/CRF/rnn_1/TensorArrayTensorArrayV3#Arguments/CRF/rnn_1/strided_slice_1*
dtype0*$
element_shape:€€€€€€€€€*
_output_shapes

:: *?
tensor_array_name*(Arguments/CRF/rnn_1/dynamic_rnn/output_0*
identical_element_shapes(
И
!Arguments/CRF/rnn_1/TensorArray_1TensorArrayV3#Arguments/CRF/rnn_1/strided_slice_1*
dtype0*$
element_shape:€€€€€€€€€*
_output_shapes

:: *
identical_element_shapes(*>
tensor_array_name)'Arguments/CRF/rnn_1/dynamic_rnn/input_0
y
,Arguments/CRF/rnn_1/TensorArrayUnstack/ShapeShapeArguments/CRF/rnn_1/transpose*
_output_shapes
:*
T0
Д
:Arguments/CRF/rnn_1/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ж
<Arguments/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ж
<Arguments/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
р
4Arguments/CRF/rnn_1/TensorArrayUnstack/strided_sliceStridedSlice,Arguments/CRF/rnn_1/TensorArrayUnstack/Shape:Arguments/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack<Arguments/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_1<Arguments/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
t
2Arguments/CRF/rnn_1/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
t
2Arguments/CRF/rnn_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ш
,Arguments/CRF/rnn_1/TensorArrayUnstack/rangeRange2Arguments/CRF/rnn_1/TensorArrayUnstack/range/start4Arguments/CRF/rnn_1/TensorArrayUnstack/strided_slice2Arguments/CRF/rnn_1/TensorArrayUnstack/range/delta*#
_output_shapes
:€€€€€€€€€
ќ
NArguments/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3!Arguments/CRF/rnn_1/TensorArray_1,Arguments/CRF/rnn_1/TensorArrayUnstack/rangeArguments/CRF/rnn_1/transpose#Arguments/CRF/rnn_1/TensorArray_1:1*0
_class&
$"loc:@Arguments/CRF/rnn_1/transpose*
T0*
_output_shapes
: 
_
Arguments/CRF/rnn_1/Maximum/xConst*
_output_shapes
: *
value	B :*
dtype0

Arguments/CRF/rnn_1/MaximumMaximumArguments/CRF/rnn_1/Maximum/xArguments/CRF/rnn_1/Max*
T0*
_output_shapes
: 
Й
Arguments/CRF/rnn_1/MinimumMinimum#Arguments/CRF/rnn_1/strided_slice_1Arguments/CRF/rnn_1/Maximum*
T0*
_output_shapes
: 
m
+Arguments/CRF/rnn_1/while/iteration_counterConst*
dtype0*
value	B : *
_output_shapes
: 
 
Arguments/CRF/rnn_1/while/EnterEnter+Arguments/CRF/rnn_1/while/iteration_counter*
T0*
_output_shapes
: *
parallel_iterations *7

frame_name)'Arguments/CRF/rnn_1/while/while_context
є
!Arguments/CRF/rnn_1/while/Enter_1EnterArguments/CRF/rnn_1/time*
T0*
_output_shapes
: *7

frame_name)'Arguments/CRF/rnn_1/while/while_context*
parallel_iterations 
¬
!Arguments/CRF/rnn_1/while/Enter_2Enter!Arguments/CRF/rnn_1/TensorArray:1*
T0*
parallel_iterations *
_output_shapes
: *7

frame_name)'Arguments/CRF/rnn_1/while/while_context
ћ
!Arguments/CRF/rnn_1/while/Enter_3EnterArguments/CRF/ExpandDims_1*'
_output_shapes
:€€€€€€€€€*7

frame_name)'Arguments/CRF/rnn_1/while/while_context*
parallel_iterations *
T0
Ю
Arguments/CRF/rnn_1/while/MergeMergeArguments/CRF/rnn_1/while/Enter'Arguments/CRF/rnn_1/while/NextIteration*
_output_shapes
: : *
N*
T0
§
!Arguments/CRF/rnn_1/while/Merge_1Merge!Arguments/CRF/rnn_1/while/Enter_1)Arguments/CRF/rnn_1/while/NextIteration_1*
N*
_output_shapes
: : *
T0
§
!Arguments/CRF/rnn_1/while/Merge_2Merge!Arguments/CRF/rnn_1/while/Enter_2)Arguments/CRF/rnn_1/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
µ
!Arguments/CRF/rnn_1/while/Merge_3Merge!Arguments/CRF/rnn_1/while/Enter_3)Arguments/CRF/rnn_1/while/NextIteration_3*
N*
T0*)
_output_shapes
:€€€€€€€€€: 
О
Arguments/CRF/rnn_1/while/LessLessArguments/CRF/rnn_1/while/Merge$Arguments/CRF/rnn_1/while/Less/Enter*
T0*
_output_shapes
: 
Џ
$Arguments/CRF/rnn_1/while/Less/EnterEnter#Arguments/CRF/rnn_1/strided_slice_1*
T0*
_output_shapes
: *
is_constant(*
parallel_iterations *7

frame_name)'Arguments/CRF/rnn_1/while/while_context
Ф
 Arguments/CRF/rnn_1/while/Less_1Less!Arguments/CRF/rnn_1/while/Merge_1&Arguments/CRF/rnn_1/while/Less_1/Enter*
T0*
_output_shapes
: 
‘
&Arguments/CRF/rnn_1/while/Less_1/EnterEnterArguments/CRF/rnn_1/Minimum*7

frame_name)'Arguments/CRF/rnn_1/while/while_context*
T0*
parallel_iterations *
_output_shapes
: *
is_constant(
М
$Arguments/CRF/rnn_1/while/LogicalAnd
LogicalAndArguments/CRF/rnn_1/while/Less Arguments/CRF/rnn_1/while/Less_1*
_output_shapes
: 
l
"Arguments/CRF/rnn_1/while/LoopCondLoopCond$Arguments/CRF/rnn_1/while/LogicalAnd*
_output_shapes
: 
∆
 Arguments/CRF/rnn_1/while/SwitchSwitchArguments/CRF/rnn_1/while/Merge"Arguments/CRF/rnn_1/while/LoopCond*
T0*2
_class(
&$loc:@Arguments/CRF/rnn_1/while/Merge*
_output_shapes
: : 
ћ
"Arguments/CRF/rnn_1/while/Switch_1Switch!Arguments/CRF/rnn_1/while/Merge_1"Arguments/CRF/rnn_1/while/LoopCond*
T0*4
_class*
(&loc:@Arguments/CRF/rnn_1/while/Merge_1*
_output_shapes
: : 
ћ
"Arguments/CRF/rnn_1/while/Switch_2Switch!Arguments/CRF/rnn_1/while/Merge_2"Arguments/CRF/rnn_1/while/LoopCond*
T0*4
_class*
(&loc:@Arguments/CRF/rnn_1/while/Merge_2*
_output_shapes
: : 
о
"Arguments/CRF/rnn_1/while/Switch_3Switch!Arguments/CRF/rnn_1/while/Merge_3"Arguments/CRF/rnn_1/while/LoopCond*
T0*4
_class*
(&loc:@Arguments/CRF/rnn_1/while/Merge_3*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€
s
"Arguments/CRF/rnn_1/while/IdentityIdentity"Arguments/CRF/rnn_1/while/Switch:1*
T0*
_output_shapes
: 
w
$Arguments/CRF/rnn_1/while/Identity_1Identity$Arguments/CRF/rnn_1/while/Switch_1:1*
T0*
_output_shapes
: 
w
$Arguments/CRF/rnn_1/while/Identity_2Identity$Arguments/CRF/rnn_1/while/Switch_2:1*
T0*
_output_shapes
: 
И
$Arguments/CRF/rnn_1/while/Identity_3Identity$Arguments/CRF/rnn_1/while/Switch_3:1*
T0*'
_output_shapes
:€€€€€€€€€
Ж
Arguments/CRF/rnn_1/while/add/yConst#^Arguments/CRF/rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
К
Arguments/CRF/rnn_1/while/addAdd"Arguments/CRF/rnn_1/while/IdentityArguments/CRF/rnn_1/while/add/y*
T0*
_output_shapes
: 
Д
+Arguments/CRF/rnn_1/while/TensorArrayReadV3TensorArrayReadV31Arguments/CRF/rnn_1/while/TensorArrayReadV3/Enter$Arguments/CRF/rnn_1/while/Identity_13Arguments/CRF/rnn_1/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:€€€€€€€€€*
dtype0
й
1Arguments/CRF/rnn_1/while/TensorArrayReadV3/EnterEnter!Arguments/CRF/rnn_1/TensorArray_1*
is_constant(*
T0*7

frame_name)'Arguments/CRF/rnn_1/while/while_context*
parallel_iterations *
_output_shapes
:
Ф
3Arguments/CRF/rnn_1/while/TensorArrayReadV3/Enter_1EnterNArguments/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
T0*7

frame_name)'Arguments/CRF/rnn_1/while/while_context*
parallel_iterations *
_output_shapes
: 
Є
&Arguments/CRF/rnn_1/while/GreaterEqualGreaterEqual$Arguments/CRF/rnn_1/while/Identity_1,Arguments/CRF/rnn_1/while/GreaterEqual/Enter*
T0*#
_output_shapes
:€€€€€€€€€
л
,Arguments/CRF/rnn_1/while/GreaterEqual/EnterEnterArguments/CRF/rnn_1/CheckSeqLen*
is_constant(*
T0*7

frame_name)'Arguments/CRF/rnn_1/while/while_context*
parallel_iterations *#
_output_shapes
:€€€€€€€€€
Ч
!Arguments/CRF/rnn_1/while/SqueezeSqueeze$Arguments/CRF/rnn_1/while/Identity_3*
T0*
squeeze_dims
*#
_output_shapes
:€€€€€€€€€
z
Arguments/CRF/rnn_1/while/ShapeShape+Arguments/CRF/rnn_1/while/TensorArrayReadV3*
T0*
_output_shapes
:
Ь
-Arguments/CRF/rnn_1/while/strided_slice/stackConst#^Arguments/CRF/rnn_1/while/Identity*
_output_shapes
:*
valueB: *
dtype0
Ю
/Arguments/CRF/rnn_1/while/strided_slice/stack_1Const#^Arguments/CRF/rnn_1/while/Identity*
_output_shapes
:*
valueB:*
dtype0
Ю
/Arguments/CRF/rnn_1/while/strided_slice/stack_2Const#^Arguments/CRF/rnn_1/while/Identity*
_output_shapes
:*
valueB:*
dtype0
ѓ
'Arguments/CRF/rnn_1/while/strided_sliceStridedSliceArguments/CRF/rnn_1/while/Shape-Arguments/CRF/rnn_1/while/strided_slice/stack/Arguments/CRF/rnn_1/while/strided_slice/stack_1/Arguments/CRF/rnn_1/while/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
М
%Arguments/CRF/rnn_1/while/range/startConst#^Arguments/CRF/rnn_1/while/Identity*
_output_shapes
: *
value	B : *
dtype0
М
%Arguments/CRF/rnn_1/while/range/deltaConst#^Arguments/CRF/rnn_1/while/Identity*
_output_shapes
: *
value	B :*
dtype0
ƒ
Arguments/CRF/rnn_1/while/rangeRange%Arguments/CRF/rnn_1/while/range/start'Arguments/CRF/rnn_1/while/strided_slice%Arguments/CRF/rnn_1/while/range/delta*#
_output_shapes
:€€€€€€€€€
≤
Arguments/CRF/rnn_1/while/stackPackArguments/CRF/rnn_1/while/range!Arguments/CRF/rnn_1/while/Squeeze*
N*'
_output_shapes
:€€€€€€€€€*

axis*
T0
ј
"Arguments/CRF/rnn_1/while/GatherNdGatherNd+Arguments/CRF/rnn_1/while/TensorArrayReadV3Arguments/CRF/rnn_1/while/stack*
Tparams0*#
_output_shapes
:€€€€€€€€€*
Tindices0
Ш
(Arguments/CRF/rnn_1/while/ExpandDims/dimConst#^Arguments/CRF/rnn_1/while/Identity*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
≤
$Arguments/CRF/rnn_1/while/ExpandDims
ExpandDims"Arguments/CRF/rnn_1/while/GatherNd(Arguments/CRF/rnn_1/while/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Л
 Arguments/CRF/rnn_1/while/SelectSelect&Arguments/CRF/rnn_1/while/GreaterEqual&Arguments/CRF/rnn_1/while/Select/Enter$Arguments/CRF/rnn_1/while/ExpandDims*7
_class-
+)loc:@Arguments/CRF/rnn_1/while/ExpandDims*'
_output_shapes
:€€€€€€€€€*
T0
Ь
&Arguments/CRF/rnn_1/while/Select/EnterEnterArguments/CRF/rnn_1/zeros*'
_output_shapes
:€€€€€€€€€*
is_constant(*
T0*
parallel_iterations *7
_class-
+)loc:@Arguments/CRF/rnn_1/while/ExpandDims*7

frame_name)'Arguments/CRF/rnn_1/while/while_context
Л
"Arguments/CRF/rnn_1/while/Select_1Select&Arguments/CRF/rnn_1/while/GreaterEqual$Arguments/CRF/rnn_1/while/Identity_3$Arguments/CRF/rnn_1/while/ExpandDims*7
_class-
+)loc:@Arguments/CRF/rnn_1/while/ExpandDims*'
_output_shapes
:€€€€€€€€€*
T0
а
=Arguments/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3CArguments/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter$Arguments/CRF/rnn_1/while/Identity_1 Arguments/CRF/rnn_1/while/Select$Arguments/CRF/rnn_1/while/Identity_2*7
_class-
+)loc:@Arguments/CRF/rnn_1/while/ExpandDims*
_output_shapes
: *
T0
≤
CArguments/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterArguments/CRF/rnn_1/TensorArray*
is_constant(*7

frame_name)'Arguments/CRF/rnn_1/while/while_context*
T0*
_output_shapes
:*7
_class-
+)loc:@Arguments/CRF/rnn_1/while/ExpandDims*
parallel_iterations 
И
!Arguments/CRF/rnn_1/while/add_1/yConst#^Arguments/CRF/rnn_1/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Р
Arguments/CRF/rnn_1/while/add_1Add$Arguments/CRF/rnn_1/while/Identity_1!Arguments/CRF/rnn_1/while/add_1/y*
T0*
_output_shapes
: 
x
'Arguments/CRF/rnn_1/while/NextIterationNextIterationArguments/CRF/rnn_1/while/add*
_output_shapes
: *
T0
|
)Arguments/CRF/rnn_1/while/NextIteration_1NextIterationArguments/CRF/rnn_1/while/add_1*
T0*
_output_shapes
: 
Ъ
)Arguments/CRF/rnn_1/while/NextIteration_2NextIteration=Arguments/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Р
)Arguments/CRF/rnn_1/while/NextIteration_3NextIteration"Arguments/CRF/rnn_1/while/Select_1*
T0*'
_output_shapes
:€€€€€€€€€
i
Arguments/CRF/rnn_1/while/ExitExit Arguments/CRF/rnn_1/while/Switch*
T0*
_output_shapes
: 
m
 Arguments/CRF/rnn_1/while/Exit_1Exit"Arguments/CRF/rnn_1/while/Switch_1*
T0*
_output_shapes
: 
m
 Arguments/CRF/rnn_1/while/Exit_2Exit"Arguments/CRF/rnn_1/while/Switch_2*
T0*
_output_shapes
: 
~
 Arguments/CRF/rnn_1/while/Exit_3Exit"Arguments/CRF/rnn_1/while/Switch_3*
T0*'
_output_shapes
:€€€€€€€€€
Џ
6Arguments/CRF/rnn_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3Arguments/CRF/rnn_1/TensorArray Arguments/CRF/rnn_1/while/Exit_2*
_output_shapes
: *2
_class(
&$loc:@Arguments/CRF/rnn_1/TensorArray
¶
0Arguments/CRF/rnn_1/TensorArrayStack/range/startConst*
_output_shapes
: *
dtype0*
value	B : *2
_class(
&$loc:@Arguments/CRF/rnn_1/TensorArray
¶
0Arguments/CRF/rnn_1/TensorArrayStack/range/deltaConst*2
_class(
&$loc:@Arguments/CRF/rnn_1/TensorArray*
dtype0*
_output_shapes
: *
value	B :
®
*Arguments/CRF/rnn_1/TensorArrayStack/rangeRange0Arguments/CRF/rnn_1/TensorArrayStack/range/start6Arguments/CRF/rnn_1/TensorArrayStack/TensorArraySizeV30Arguments/CRF/rnn_1/TensorArrayStack/range/delta*2
_class(
&$loc:@Arguments/CRF/rnn_1/TensorArray*#
_output_shapes
:€€€€€€€€€
”
8Arguments/CRF/rnn_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3Arguments/CRF/rnn_1/TensorArray*Arguments/CRF/rnn_1/TensorArrayStack/range Arguments/CRF/rnn_1/while/Exit_2*2
_class(
&$loc:@Arguments/CRF/rnn_1/TensorArray*$
element_shape:€€€€€€€€€*
dtype0*,
_output_shapes
:Ю€€€€€€€€€
e
Arguments/CRF/rnn_1/Const_4Const*
dtype0*
_output_shapes
:*
valueB:
\
Arguments/CRF/rnn_1/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
c
!Arguments/CRF/rnn_1/range_1/startConst*
dtype0*
_output_shapes
: *
value	B :
c
!Arguments/CRF/rnn_1/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Ґ
Arguments/CRF/rnn_1/range_1Range!Arguments/CRF/rnn_1/range_1/startArguments/CRF/rnn_1/Rank_1!Arguments/CRF/rnn_1/range_1/delta*
_output_shapes
:
v
%Arguments/CRF/rnn_1/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
c
!Arguments/CRF/rnn_1/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
љ
Arguments/CRF/rnn_1/concat_2ConcatV2%Arguments/CRF/rnn_1/concat_2/values_0Arguments/CRF/rnn_1/range_1!Arguments/CRF/rnn_1/concat_2/axis*
_output_shapes
:*
T0*
N
ї
Arguments/CRF/rnn_1/transpose_1	Transpose8Arguments/CRF/rnn_1/TensorArrayStack/TensorArrayGatherV3Arguments/CRF/rnn_1/concat_2*,
_output_shapes
:€€€€€€€€€Ю*
T0
Н
Arguments/CRF/Squeeze_1SqueezeArguments/CRF/rnn_1/transpose_1*(
_output_shapes
:€€€€€€€€€Ю*
T0*
squeeze_dims

[
Arguments/CRF/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
ђ
Arguments/CRF/concatConcatV2Arguments/CRF/ExpandDims_1Arguments/CRF/Squeeze_1Arguments/CRF/concat/axis*(
_output_shapes
:€€€€€€€€€Я*
T0*
N
µ
Arguments/CRF/ReverseSequence_1ReverseSequenceArguments/CRF/concatarg_original_sequence_lengths*(
_output_shapes
:€€€€€€€€€Я*

Tlen0*
seq_dim*
T0
e
#Arguments/CRF/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
Л
Arguments/CRF/MaxMaxArguments/CRF/rnn/while/Exit_3#Arguments/CRF/Max/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0
|
Arguments/ToInt64CastArguments/CRF/ReverseSequence_1*

DstT0	*(
_output_shapes
:€€€€€€€€€Я*

SrcT0
Ё
-Arguments/hash_table_Lookup/LookupTableFindV2LookupTableFindV2$Arguments/index_to_string/hash_tableArguments/ToInt64Arguments/index_to_string/Const*(
_output_shapes
:€€€€€€€€€Я*	
Tin0	*

Tout0
^
Arguments/viterbiIdentityArguments/CRF/Max*#
_output_shapes
:€€€€€€€€€*
T0

initNoOp
\
init_all_tablesNoOpC^Arguments/index_to_string/table_init/InitializeTableFromTextFileV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
Д
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_3357e35ab2be42c29c7775b17cd6c063/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
√
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*з
valueЁBЏB"Arguments/BiLSTM/bw/lstm_cell/biasB$Arguments/BiLSTM/bw/lstm_cell/kernelB"Arguments/BiLSTM/fw/lstm_cell/biasB$Arguments/BiLSTM/fw/lstm_cell/kernelBArguments/CRF/transitionBArguments/arg_mask/VariableBArguments/dep/VariableBArguments/pos/VariableBArguments/proj_1/W_1BArguments/proj_1/b_1BArguments/prop_mask/VariableBglobal_step
К
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*+
value"B B B B B B B B B B B B B 
а
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"Arguments/BiLSTM/bw/lstm_cell/bias$Arguments/BiLSTM/bw/lstm_cell/kernel"Arguments/BiLSTM/fw/lstm_cell/bias$Arguments/BiLSTM/fw/lstm_cell/kernelArguments/CRF/transitionArguments/arg_mask/VariableArguments/dep/VariableArguments/pos/VariableArguments/proj_1/W_1Arguments/proj_1/b_1Arguments/prop_mask/Variableglobal_step"/device:CPU:0*
dtypes
2	
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
†
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
_output_shapes
:*
N*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
∆
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*з
valueЁBЏB"Arguments/BiLSTM/bw/lstm_cell/biasB$Arguments/BiLSTM/bw/lstm_cell/kernelB"Arguments/BiLSTM/fw/lstm_cell/biasB$Arguments/BiLSTM/fw/lstm_cell/kernelBArguments/CRF/transitionBArguments/arg_mask/VariableBArguments/dep/VariableBArguments/pos/VariableBArguments/proj_1/W_1BArguments/proj_1/b_1BArguments/prop_mask/VariableBglobal_step
Н
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*+
value"B B B B B B B B B B B B B *
dtype0
÷
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	
¶
save/AssignAssign"Arguments/BiLSTM/bw/lstm_cell/biassave/RestoreV2*
_output_shapes	
:†*5
_class+
)'loc:@Arguments/BiLSTM/bw/lstm_cell/bias*
T0
≥
save/Assign_1Assign$Arguments/BiLSTM/bw/lstm_cell/kernelsave/RestoreV2:1* 
_output_shapes
:
Ш†*7
_class-
+)loc:@Arguments/BiLSTM/bw/lstm_cell/kernel*
T0
™
save/Assign_2Assign"Arguments/BiLSTM/fw/lstm_cell/biassave/RestoreV2:2*5
_class+
)'loc:@Arguments/BiLSTM/fw/lstm_cell/bias*
_output_shapes	
:†*
T0
≥
save/Assign_3Assign$Arguments/BiLSTM/fw/lstm_cell/kernelsave/RestoreV2:3* 
_output_shapes
:
Ш†*7
_class-
+)loc:@Arguments/BiLSTM/fw/lstm_cell/kernel*
T0
Щ
save/Assign_4AssignArguments/CRF/transitionsave/RestoreV2:4*
_output_shapes

:*+
_class!
loc:@Arguments/CRF/transition*
T0
Я
save/Assign_5AssignArguments/arg_mask/Variablesave/RestoreV2:5*.
_class$
" loc:@Arguments/arg_mask/Variable*
_output_shapes

:*
T0
Х
save/Assign_6AssignArguments/dep/Variablesave/RestoreV2:6*)
_class
loc:@Arguments/dep/Variable*
_output_shapes

:>*
T0
Х
save/Assign_7AssignArguments/pos/Variablesave/RestoreV2:7*)
_class
loc:@Arguments/pos/Variable*
_output_shapes

:*
T0
Т
save/Assign_8AssignArguments/proj_1/W_1save/RestoreV2:8*'
_class
loc:@Arguments/proj_1/W_1*
_output_shapes
:	»*
T0
Н
save/Assign_9AssignArguments/proj_1/b_1save/RestoreV2:9*'
_class
loc:@Arguments/proj_1/b_1*
_output_shapes
:*
T0
£
save/Assign_10AssignArguments/prop_mask/Variablesave/RestoreV2:10*/
_class%
#!loc:@Arguments/prop_mask/Variable*
_output_shapes

:*
T0
y
save/Assign_11Assignglobal_stepsave/RestoreV2:11*
_class
loc:@global_step*
_output_shapes
: *
T0	
Џ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"ЩЩ
while_contextЖЩВЩ
а-
1Arguments/BiLSTM/BiLSTM/fw/fw/while/while_context *.Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCond:02+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge:0:.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity:0B*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit:0B,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_1:0B,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_2:0B,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_3:0B,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_4:0JЫ(
+Arguments/BiLSTM/BiLSTM/fw/fw/CheckSeqLen:0
'Arguments/BiLSTM/BiLSTM/fw/fw/Minimum:0
+Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray:0
ZArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
-Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray_1:0
/Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1:0
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_1:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_2:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_3:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_4:0
*Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_1:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_2:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_3:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Exit_4:0
8Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Enter:0
2Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity:0
0Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_1:0
0Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_2:0
0Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_3:0
0Arguments/BiLSTM/BiLSTM/fw/fw/while/Identity_4:0
0Arguments/BiLSTM/BiLSTM/fw/fw/while/Less/Enter:0
*Arguments/BiLSTM/BiLSTM/fw/fw/while/Less:0
2Arguments/BiLSTM/BiLSTM/fw/fw/while/Less_1/Enter:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Less_1:0
0Arguments/BiLSTM/BiLSTM/fw/fw/while/LogicalAnd:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/LoopCond:0
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge:0
+Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge:1
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_1:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_1:1
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_2:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_2:1
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_3:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_3:1
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_4:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/Merge_4:1
3Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration:0
5Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_1:0
5Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_2:0
5Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_3:0
5Arguments/BiLSTM/BiLSTM/fw/fw/while/NextIteration_4:0
2Arguments/BiLSTM/BiLSTM/fw/fw/while/Select/Enter:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Select:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Select_1:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Select_2:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch:0
,Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch:1
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_1:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_1:1
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_2:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_2:1
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_3:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_3:1
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_4:0
.Arguments/BiLSTM/BiLSTM/fw/fw/while/Switch_4:1
=Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter:0
?Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1:0
7Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3:0
OArguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
IArguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
+Arguments/BiLSTM/BiLSTM/fw/fw/while/add/y:0
)Arguments/BiLSTM/BiLSTM/fw/fw/while/add:0
-Arguments/BiLSTM/BiLSTM/fw/fw/while/add_1/y:0
+Arguments/BiLSTM/BiLSTM/fw/fw/while/add_1:0
CArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter:0
EArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1:0
=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:0
=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:1
=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:2
=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:3
=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:4
=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:5
=Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:6
5Arguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zeros:0
%Arguments/BiLSTM/BiLSTM/fw/fw/zeros:0
)Arguments/BiLSTM/fw/lstm_cell/bias/read:0
+Arguments/BiLSTM/fw/lstm_cell/kernel/read:0g
+Arguments/BiLSTM/BiLSTM/fw/fw/CheckSeqLen:08Arguments/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Enter:0r
)Arguments/BiLSTM/fw/lstm_cell/bias/read:0EArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1:0n
-Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray_1:0=Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter:0Э
ZArguments/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0?Arguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1:0c
/Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1:00Arguments/BiLSTM/BiLSTM/fw/fw/while/Less/Enter:0]
'Arguments/BiLSTM/BiLSTM/fw/fw/Minimum:02Arguments/BiLSTM/BiLSTM/fw/fw/while/Less_1/Enter:0r
+Arguments/BiLSTM/fw/lstm_cell/kernel/read:0CArguments/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter:0[
%Arguments/BiLSTM/BiLSTM/fw/fw/zeros:02Arguments/BiLSTM/BiLSTM/fw/fw/while/Select/Enter:0~
+Arguments/BiLSTM/BiLSTM/fw/fw/TensorArray:0OArguments/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0R+Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter:0R-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_1:0R-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_2:0R-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_3:0R-Arguments/BiLSTM/BiLSTM/fw/fw/while/Enter_4:0Z/Arguments/BiLSTM/BiLSTM/fw/fw/strided_slice_1:0
а-
1Arguments/BiLSTM/BiLSTM/bw/bw/while/while_context *.Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCond:02+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge:0:.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity:0B*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit:0B,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_1:0B,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_2:0B,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_3:0B,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_4:0JЫ(
+Arguments/BiLSTM/BiLSTM/bw/bw/CheckSeqLen:0
'Arguments/BiLSTM/BiLSTM/bw/bw/Minimum:0
+Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray:0
ZArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
-Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray_1:0
/Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1:0
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_1:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_2:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_3:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_4:0
*Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_1:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_2:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_3:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Exit_4:0
8Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Enter:0
2Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity:0
0Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_1:0
0Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_2:0
0Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_3:0
0Arguments/BiLSTM/BiLSTM/bw/bw/while/Identity_4:0
0Arguments/BiLSTM/BiLSTM/bw/bw/while/Less/Enter:0
*Arguments/BiLSTM/BiLSTM/bw/bw/while/Less:0
2Arguments/BiLSTM/BiLSTM/bw/bw/while/Less_1/Enter:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Less_1:0
0Arguments/BiLSTM/BiLSTM/bw/bw/while/LogicalAnd:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/LoopCond:0
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge:0
+Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge:1
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_1:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_1:1
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_2:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_2:1
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_3:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_3:1
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_4:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/Merge_4:1
3Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration:0
5Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_1:0
5Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_2:0
5Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_3:0
5Arguments/BiLSTM/BiLSTM/bw/bw/while/NextIteration_4:0
2Arguments/BiLSTM/BiLSTM/bw/bw/while/Select/Enter:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Select:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Select_1:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Select_2:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch:0
,Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch:1
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_1:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_1:1
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_2:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_2:1
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_3:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_3:1
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_4:0
.Arguments/BiLSTM/BiLSTM/bw/bw/while/Switch_4:1
=Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter:0
?Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1:0
7Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3:0
OArguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
IArguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
+Arguments/BiLSTM/BiLSTM/bw/bw/while/add/y:0
)Arguments/BiLSTM/BiLSTM/bw/bw/while/add:0
-Arguments/BiLSTM/BiLSTM/bw/bw/while/add_1/y:0
+Arguments/BiLSTM/BiLSTM/bw/bw/while/add_1:0
CArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter:0
EArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1:0
=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:0
=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:1
=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:2
=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:3
=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:4
=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:5
=Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:6
5Arguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zeros:0
%Arguments/BiLSTM/BiLSTM/bw/bw/zeros:0
)Arguments/BiLSTM/bw/lstm_cell/bias/read:0
+Arguments/BiLSTM/bw/lstm_cell/kernel/read:0c
/Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1:00Arguments/BiLSTM/BiLSTM/bw/bw/while/Less/Enter:0r
+Arguments/BiLSTM/bw/lstm_cell/kernel/read:0CArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter:0g
+Arguments/BiLSTM/BiLSTM/bw/bw/CheckSeqLen:08Arguments/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Enter:0[
%Arguments/BiLSTM/BiLSTM/bw/bw/zeros:02Arguments/BiLSTM/BiLSTM/bw/bw/while/Select/Enter:0n
-Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray_1:0=Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter:0~
+Arguments/BiLSTM/BiLSTM/bw/bw/TensorArray:0OArguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Э
ZArguments/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0?Arguments/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1:0]
'Arguments/BiLSTM/BiLSTM/bw/bw/Minimum:02Arguments/BiLSTM/BiLSTM/bw/bw/while/Less_1/Enter:0r
)Arguments/BiLSTM/bw/lstm_cell/bias/read:0EArguments/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1:0R+Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter:0R-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_1:0R-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_2:0R-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_3:0R-Arguments/BiLSTM/BiLSTM/bw/bw/while/Enter_4:0Z/Arguments/BiLSTM/BiLSTM/bw/bw/strided_slice_1:0
г
%Arguments/CRF/rnn/while/while_context *"Arguments/CRF/rnn/while/LoopCond:02Arguments/CRF/rnn/while/Merge:0:"Arguments/CRF/rnn/while/Identity:0BArguments/CRF/rnn/while/Exit:0B Arguments/CRF/rnn/while/Exit_1:0B Arguments/CRF/rnn/while/Exit_2:0B Arguments/CRF/rnn/while/Exit_3:0JЧ
Arguments/CRF/ExpandDims:0
Arguments/CRF/rnn/CheckSeqLen:0
Arguments/CRF/rnn/Minimum:0
Arguments/CRF/rnn/TensorArray:0
NArguments/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
!Arguments/CRF/rnn/TensorArray_1:0
#Arguments/CRF/rnn/strided_slice_1:0
*Arguments/CRF/rnn/while/ArgMax/dimension:0
 Arguments/CRF/rnn/while/ArgMax:0
Arguments/CRF/rnn/while/Cast:0
Arguments/CRF/rnn/while/Enter:0
!Arguments/CRF/rnn/while/Enter_1:0
!Arguments/CRF/rnn/while/Enter_2:0
!Arguments/CRF/rnn/while/Enter_3:0
Arguments/CRF/rnn/while/Exit:0
 Arguments/CRF/rnn/while/Exit_1:0
 Arguments/CRF/rnn/while/Exit_2:0
 Arguments/CRF/rnn/while/Exit_3:0
(Arguments/CRF/rnn/while/ExpandDims/dim:0
$Arguments/CRF/rnn/while/ExpandDims:0
,Arguments/CRF/rnn/while/GreaterEqual/Enter:0
&Arguments/CRF/rnn/while/GreaterEqual:0
"Arguments/CRF/rnn/while/Identity:0
$Arguments/CRF/rnn/while/Identity_1:0
$Arguments/CRF/rnn/while/Identity_2:0
$Arguments/CRF/rnn/while/Identity_3:0
$Arguments/CRF/rnn/while/Less/Enter:0
Arguments/CRF/rnn/while/Less:0
&Arguments/CRF/rnn/while/Less_1/Enter:0
 Arguments/CRF/rnn/while/Less_1:0
$Arguments/CRF/rnn/while/LogicalAnd:0
"Arguments/CRF/rnn/while/LoopCond:0
/Arguments/CRF/rnn/while/Max/reduction_indices:0
Arguments/CRF/rnn/while/Max:0
Arguments/CRF/rnn/while/Merge:0
Arguments/CRF/rnn/while/Merge:1
!Arguments/CRF/rnn/while/Merge_1:0
!Arguments/CRF/rnn/while/Merge_1:1
!Arguments/CRF/rnn/while/Merge_2:0
!Arguments/CRF/rnn/while/Merge_2:1
!Arguments/CRF/rnn/while/Merge_3:0
!Arguments/CRF/rnn/while/Merge_3:1
'Arguments/CRF/rnn/while/NextIteration:0
)Arguments/CRF/rnn/while/NextIteration_1:0
)Arguments/CRF/rnn/while/NextIteration_2:0
)Arguments/CRF/rnn/while/NextIteration_3:0
&Arguments/CRF/rnn/while/Select/Enter:0
 Arguments/CRF/rnn/while/Select:0
"Arguments/CRF/rnn/while/Select_1:0
 Arguments/CRF/rnn/while/Switch:0
 Arguments/CRF/rnn/while/Switch:1
"Arguments/CRF/rnn/while/Switch_1:0
"Arguments/CRF/rnn/while/Switch_1:1
"Arguments/CRF/rnn/while/Switch_2:0
"Arguments/CRF/rnn/while/Switch_2:1
"Arguments/CRF/rnn/while/Switch_3:0
"Arguments/CRF/rnn/while/Switch_3:1
1Arguments/CRF/rnn/while/TensorArrayReadV3/Enter:0
3Arguments/CRF/rnn/while/TensorArrayReadV3/Enter_1:0
+Arguments/CRF/rnn/while/TensorArrayReadV3:0
CArguments/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
=Arguments/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
Arguments/CRF/rnn/while/add/y:0
Arguments/CRF/rnn/while/add:0
%Arguments/CRF/rnn/while/add_1/Enter:0
Arguments/CRF/rnn/while/add_1:0
Arguments/CRF/rnn/while/add_2:0
!Arguments/CRF/rnn/while/add_3/y:0
Arguments/CRF/rnn/while/add_3:0
Arguments/CRF/rnn/zeros:0C
Arguments/CRF/ExpandDims:0%Arguments/CRF/rnn/while/add_1/Enter:0O
Arguments/CRF/rnn/CheckSeqLen:0,Arguments/CRF/rnn/while/GreaterEqual/Enter:0E
Arguments/CRF/rnn/Minimum:0&Arguments/CRF/rnn/while/Less_1/Enter:0f
Arguments/CRF/rnn/TensorArray:0CArguments/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0C
Arguments/CRF/rnn/zeros:0&Arguments/CRF/rnn/while/Select/Enter:0V
!Arguments/CRF/rnn/TensorArray_1:01Arguments/CRF/rnn/while/TensorArrayReadV3/Enter:0Е
NArguments/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:03Arguments/CRF/rnn/while/TensorArrayReadV3/Enter_1:0K
#Arguments/CRF/rnn/strided_slice_1:0$Arguments/CRF/rnn/while/Less/Enter:0RArguments/CRF/rnn/while/Enter:0R!Arguments/CRF/rnn/while/Enter_1:0R!Arguments/CRF/rnn/while/Enter_2:0R!Arguments/CRF/rnn/while/Enter_3:0Z#Arguments/CRF/rnn/strided_slice_1:0
”
'Arguments/CRF/rnn_1/while/while_context *$Arguments/CRF/rnn_1/while/LoopCond:02!Arguments/CRF/rnn_1/while/Merge:0:$Arguments/CRF/rnn_1/while/Identity:0B Arguments/CRF/rnn_1/while/Exit:0B"Arguments/CRF/rnn_1/while/Exit_1:0B"Arguments/CRF/rnn_1/while/Exit_2:0B"Arguments/CRF/rnn_1/while/Exit_3:0Jн
!Arguments/CRF/rnn_1/CheckSeqLen:0
Arguments/CRF/rnn_1/Minimum:0
!Arguments/CRF/rnn_1/TensorArray:0
PArguments/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
#Arguments/CRF/rnn_1/TensorArray_1:0
%Arguments/CRF/rnn_1/strided_slice_1:0
!Arguments/CRF/rnn_1/while/Enter:0
#Arguments/CRF/rnn_1/while/Enter_1:0
#Arguments/CRF/rnn_1/while/Enter_2:0
#Arguments/CRF/rnn_1/while/Enter_3:0
 Arguments/CRF/rnn_1/while/Exit:0
"Arguments/CRF/rnn_1/while/Exit_1:0
"Arguments/CRF/rnn_1/while/Exit_2:0
"Arguments/CRF/rnn_1/while/Exit_3:0
*Arguments/CRF/rnn_1/while/ExpandDims/dim:0
&Arguments/CRF/rnn_1/while/ExpandDims:0
$Arguments/CRF/rnn_1/while/GatherNd:0
.Arguments/CRF/rnn_1/while/GreaterEqual/Enter:0
(Arguments/CRF/rnn_1/while/GreaterEqual:0
$Arguments/CRF/rnn_1/while/Identity:0
&Arguments/CRF/rnn_1/while/Identity_1:0
&Arguments/CRF/rnn_1/while/Identity_2:0
&Arguments/CRF/rnn_1/while/Identity_3:0
&Arguments/CRF/rnn_1/while/Less/Enter:0
 Arguments/CRF/rnn_1/while/Less:0
(Arguments/CRF/rnn_1/while/Less_1/Enter:0
"Arguments/CRF/rnn_1/while/Less_1:0
&Arguments/CRF/rnn_1/while/LogicalAnd:0
$Arguments/CRF/rnn_1/while/LoopCond:0
!Arguments/CRF/rnn_1/while/Merge:0
!Arguments/CRF/rnn_1/while/Merge:1
#Arguments/CRF/rnn_1/while/Merge_1:0
#Arguments/CRF/rnn_1/while/Merge_1:1
#Arguments/CRF/rnn_1/while/Merge_2:0
#Arguments/CRF/rnn_1/while/Merge_2:1
#Arguments/CRF/rnn_1/while/Merge_3:0
#Arguments/CRF/rnn_1/while/Merge_3:1
)Arguments/CRF/rnn_1/while/NextIteration:0
+Arguments/CRF/rnn_1/while/NextIteration_1:0
+Arguments/CRF/rnn_1/while/NextIteration_2:0
+Arguments/CRF/rnn_1/while/NextIteration_3:0
(Arguments/CRF/rnn_1/while/Select/Enter:0
"Arguments/CRF/rnn_1/while/Select:0
$Arguments/CRF/rnn_1/while/Select_1:0
!Arguments/CRF/rnn_1/while/Shape:0
#Arguments/CRF/rnn_1/while/Squeeze:0
"Arguments/CRF/rnn_1/while/Switch:0
"Arguments/CRF/rnn_1/while/Switch:1
$Arguments/CRF/rnn_1/while/Switch_1:0
$Arguments/CRF/rnn_1/while/Switch_1:1
$Arguments/CRF/rnn_1/while/Switch_2:0
$Arguments/CRF/rnn_1/while/Switch_2:1
$Arguments/CRF/rnn_1/while/Switch_3:0
$Arguments/CRF/rnn_1/while/Switch_3:1
3Arguments/CRF/rnn_1/while/TensorArrayReadV3/Enter:0
5Arguments/CRF/rnn_1/while/TensorArrayReadV3/Enter_1:0
-Arguments/CRF/rnn_1/while/TensorArrayReadV3:0
EArguments/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
?Arguments/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3:0
!Arguments/CRF/rnn_1/while/add/y:0
Arguments/CRF/rnn_1/while/add:0
#Arguments/CRF/rnn_1/while/add_1/y:0
!Arguments/CRF/rnn_1/while/add_1:0
'Arguments/CRF/rnn_1/while/range/delta:0
'Arguments/CRF/rnn_1/while/range/start:0
!Arguments/CRF/rnn_1/while/range:0
!Arguments/CRF/rnn_1/while/stack:0
/Arguments/CRF/rnn_1/while/strided_slice/stack:0
1Arguments/CRF/rnn_1/while/strided_slice/stack_1:0
1Arguments/CRF/rnn_1/while/strided_slice/stack_2:0
)Arguments/CRF/rnn_1/while/strided_slice:0
Arguments/CRF/rnn_1/zeros:0S
!Arguments/CRF/rnn_1/CheckSeqLen:0.Arguments/CRF/rnn_1/while/GreaterEqual/Enter:0G
Arguments/CRF/rnn_1/zeros:0(Arguments/CRF/rnn_1/while/Select/Enter:0I
Arguments/CRF/rnn_1/Minimum:0(Arguments/CRF/rnn_1/while/Less_1/Enter:0O
%Arguments/CRF/rnn_1/strided_slice_1:0&Arguments/CRF/rnn_1/while/Less/Enter:0Й
PArguments/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:05Arguments/CRF/rnn_1/while/TensorArrayReadV3/Enter_1:0j
!Arguments/CRF/rnn_1/TensorArray:0EArguments/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Z
#Arguments/CRF/rnn_1/TensorArray_1:03Arguments/CRF/rnn_1/while/TensorArrayReadV3/Enter:0R!Arguments/CRF/rnn_1/while/Enter:0R#Arguments/CRF/rnn_1/while/Enter_1:0R#Arguments/CRF/rnn_1/while/Enter_2:0R#Arguments/CRF/rnn_1/while/Enter_3:0Z%Arguments/CRF/rnn_1/strided_slice_1:0"[
table_initializerF
D
BArguments/index_to_string/table_init/InitializeTableFromTextFileV2"=
asset_filepaths*
(
&Arguments/index_to_string/asset_path:0"В
saved_model_assetsl*j
h
+type.googleapis.com/tensorflow.AssetFileDef9
(
&Arguments/index_to_string/asset_path:0index_tag.txt"Ђ
	variablesЭЪ
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
z
Arguments/pos/Variable:0Arguments/pos/Variable/AssignArguments/pos/Variable/read:02Arguments/pos/random_uniform:08
z
Arguments/dep/Variable:0Arguments/dep/Variable/AssignArguments/dep/Variable/read:02Arguments/dep/random_uniform:08
О
Arguments/arg_mask/Variable:0"Arguments/arg_mask/Variable/Assign"Arguments/arg_mask/Variable/read:02#Arguments/arg_mask/random_uniform:08
Т
Arguments/prop_mask/Variable:0#Arguments/prop_mask/Variable/Assign#Arguments/prop_mask/Variable/read:02$Arguments/prop_mask/random_uniform:08
«
&Arguments/BiLSTM/fw/lstm_cell/kernel:0+Arguments/BiLSTM/fw/lstm_cell/kernel/Assign+Arguments/BiLSTM/fw/lstm_cell/kernel/read:02AArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform:08
ґ
$Arguments/BiLSTM/fw/lstm_cell/bias:0)Arguments/BiLSTM/fw/lstm_cell/bias/Assign)Arguments/BiLSTM/fw/lstm_cell/bias/read:026Arguments/BiLSTM/fw/lstm_cell/bias/Initializer/Const:08
«
&Arguments/BiLSTM/bw/lstm_cell/kernel:0+Arguments/BiLSTM/bw/lstm_cell/kernel/Assign+Arguments/BiLSTM/bw/lstm_cell/kernel/read:02AArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform:08
ґ
$Arguments/BiLSTM/bw/lstm_cell/bias:0)Arguments/BiLSTM/bw/lstm_cell/bias/Assign)Arguments/BiLSTM/bw/lstm_cell/bias/read:026Arguments/BiLSTM/bw/lstm_cell/bias/Initializer/Const:08
~
Arguments/proj_1/W_1:0Arguments/proj_1/W_1/AssignArguments/proj_1/W_1/read:02(Arguments/proj_1/W_1/Initializer/mul_1:08
~
Arguments/proj_1/b_1:0Arguments/proj_1/b_1/AssignArguments/proj_1/b_1/read:02(Arguments/proj_1/b_1/Initializer/zeros:08
Ч
Arguments/CRF/transition:0Arguments/CRF/transition/AssignArguments/CRF/transition/read:025Arguments/CRF/transition/Initializer/random_uniform:08"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"ў
trainable_variablesЅЊ
z
Arguments/pos/Variable:0Arguments/pos/Variable/AssignArguments/pos/Variable/read:02Arguments/pos/random_uniform:08
z
Arguments/dep/Variable:0Arguments/dep/Variable/AssignArguments/dep/Variable/read:02Arguments/dep/random_uniform:08
О
Arguments/arg_mask/Variable:0"Arguments/arg_mask/Variable/Assign"Arguments/arg_mask/Variable/read:02#Arguments/arg_mask/random_uniform:08
Т
Arguments/prop_mask/Variable:0#Arguments/prop_mask/Variable/Assign#Arguments/prop_mask/Variable/read:02$Arguments/prop_mask/random_uniform:08
«
&Arguments/BiLSTM/fw/lstm_cell/kernel:0+Arguments/BiLSTM/fw/lstm_cell/kernel/Assign+Arguments/BiLSTM/fw/lstm_cell/kernel/read:02AArguments/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform:08
ґ
$Arguments/BiLSTM/fw/lstm_cell/bias:0)Arguments/BiLSTM/fw/lstm_cell/bias/Assign)Arguments/BiLSTM/fw/lstm_cell/bias/read:026Arguments/BiLSTM/fw/lstm_cell/bias/Initializer/Const:08
«
&Arguments/BiLSTM/bw/lstm_cell/kernel:0+Arguments/BiLSTM/bw/lstm_cell/kernel/Assign+Arguments/BiLSTM/bw/lstm_cell/kernel/read:02AArguments/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform:08
ґ
$Arguments/BiLSTM/bw/lstm_cell/bias:0)Arguments/BiLSTM/bw/lstm_cell/bias/Assign)Arguments/BiLSTM/bw/lstm_cell/bias/read:026Arguments/BiLSTM/bw/lstm_cell/bias/Initializer/Const:08
~
Arguments/proj_1/W_1:0Arguments/proj_1/W_1/AssignArguments/proj_1/W_1/read:02(Arguments/proj_1/W_1/Initializer/mul_1:08
~
Arguments/proj_1/b_1:0Arguments/proj_1/b_1/AssignArguments/proj_1/b_1/read:02(Arguments/proj_1/b_1/Initializer/zeros:08
Ч
Arguments/CRF/transition:0Arguments/CRF/transition/AssignArguments/CRF/transition/read:025Arguments/CRF/transition/Initializer/random_uniform:08*ч
predictл
6
arg_mask*

arg_mask:0€€€€€€€€€€€€€€€€€€
8
	arg_input+
arg_input:0€€€€€€€€€€€€€€€€€€
8
	prop_mask+
prop_mask:0€€€€€€€€€€€€€€€€€€
4
arg_dep)
	arg_dep:0€€€€€€€€€€€€€€€€€€
S
arg_original_sequence_lengths2
arg_original_sequence_lengths:0€€€€€€€€€O
tagsG
/Arguments/hash_table_Lookup/LookupTableFindV2:0€€€€€€€€€Я7
probabilities&
Arguments/viterbi:0€€€€€€€€€F
sequence_lenghts2
arg_original_sequence_lengths:0€€€€€€€€€D
classes9
!Arguments/CRF/ReverseSequence_1:0€€€€€€€€€Яtensorflow/serving/predict*€
serving_defaultл
6
arg_mask*

arg_mask:0€€€€€€€€€€€€€€€€€€
8
	arg_input+
arg_input:0€€€€€€€€€€€€€€€€€€
8
	prop_mask+
prop_mask:0€€€€€€€€€€€€€€€€€€
4
arg_dep)
	arg_dep:0€€€€€€€€€€€€€€€€€€
S
arg_original_sequence_lengths2
arg_original_sequence_lengths:0€€€€€€€€€O
tagsG
/Arguments/hash_table_Lookup/LookupTableFindV2:0€€€€€€€€€Я7
probabilities&
Arguments/viterbi:0€€€€€€€€€F
sequence_lenghts2
arg_original_sequence_lengths:0€€€€€€€€€D
classes9
!Arguments/CRF/ReverseSequence_1:0€€€€€€€€€Яtensorflow/serving/predict