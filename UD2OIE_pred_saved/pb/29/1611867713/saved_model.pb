ȉ
?2?2
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
?
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
	summarizeint?
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
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
?
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
?
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
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
?
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0?????????"
value_indexint(0?????????"+

vocab_sizeint?????????(0?????????"
	delimiterstring	?
?
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
forget_biasfloat%  ??"
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
?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
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
?
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
?
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
delete_old_dirsbool(?
?
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
2	?
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
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
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
list(type)(0?
?
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
list(type)(0?
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
?
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
element_shapeshape:?
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype?
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype?
9
TensorArraySizeV3

handle
flow_in
size?
?
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring ?
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype?
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
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.14.02unknown8??

global_step/Initializer/zerosConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
k
global_step
VariableV2*
dtype0	*
_class
loc:@global_step*
shape: *
_output_shapes
: 
?
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


pred_inputPlaceholder*%
shape:??????????????????*
dtype0*0
_output_shapes
:??????????????????
~
	pred_maskPlaceholder*%
shape:??????????????????*
dtype0*0
_output_shapes
:??????????????????
y
pred_original_sequence_lengthsPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
}
pred_depPlaceholder*0
_output_shapes
:??????????????????*%
shape:??????????????????*
dtype0


labels_arcPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
t
#Predicates/pos/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
f
!Predicates/pos/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
f
!Predicates/pos/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
+Predicates/pos/random_uniform/RandomUniformRandomUniform#Predicates/pos/random_uniform/shape*
_output_shapes

:*
dtype0*
T0
?
!Predicates/pos/random_uniform/subSub!Predicates/pos/random_uniform/max!Predicates/pos/random_uniform/min*
_output_shapes
: *
T0
?
!Predicates/pos/random_uniform/mulMul+Predicates/pos/random_uniform/RandomUniform!Predicates/pos/random_uniform/sub*
_output_shapes

:*
T0
?
Predicates/pos/random_uniformAdd!Predicates/pos/random_uniform/mul!Predicates/pos/random_uniform/min*
_output_shapes

:*
T0
g
Predicates/pos/Variable
VariableV2*
_output_shapes

:*
shape
:*
dtype0
?
Predicates/pos/Variable/AssignAssignPredicates/pos/VariablePredicates/pos/random_uniform*
T0*
_output_shapes

:**
_class 
loc:@Predicates/pos/Variable
?
Predicates/pos/Variable/readIdentityPredicates/pos/Variable*
T0*
_output_shapes

:**
_class 
loc:@Predicates/pos/Variable
?
$Predicates/pos/embedding_lookup/axisConst*
value	B : *
_output_shapes
: **
_class 
loc:@Predicates/pos/Variable*
dtype0
?
Predicates/pos/embedding_lookupGatherV2Predicates/pos/Variable/read
pred_input$Predicates/pos/embedding_lookup/axis*
Tindices0*4
_output_shapes"
 :??????????????????*
Tparams0**
_class 
loc:@Predicates/pos/Variable*
Taxis0
?
(Predicates/pos/embedding_lookup/IdentityIdentityPredicates/pos/embedding_lookup*
T0*4
_output_shapes"
 :??????????????????
t
#Predicates/dep/random_uniform/shapeConst*
valueB">      *
_output_shapes
:*
dtype0
f
!Predicates/dep/random_uniform/minConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
f
!Predicates/dep/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
+Predicates/dep/random_uniform/RandomUniformRandomUniform#Predicates/dep/random_uniform/shape*
_output_shapes

:>*
dtype0*
T0
?
!Predicates/dep/random_uniform/subSub!Predicates/dep/random_uniform/max!Predicates/dep/random_uniform/min*
_output_shapes
: *
T0
?
!Predicates/dep/random_uniform/mulMul+Predicates/dep/random_uniform/RandomUniform!Predicates/dep/random_uniform/sub*
_output_shapes

:>*
T0
?
Predicates/dep/random_uniformAdd!Predicates/dep/random_uniform/mul!Predicates/dep/random_uniform/min*
_output_shapes

:>*
T0
g
Predicates/dep/Variable
VariableV2*
_output_shapes

:>*
dtype0*
shape
:>
?
Predicates/dep/Variable/AssignAssignPredicates/dep/VariablePredicates/dep/random_uniform*
_output_shapes

:>**
_class 
loc:@Predicates/dep/Variable*
T0
?
Predicates/dep/Variable/readIdentityPredicates/dep/Variable*
_output_shapes

:>**
_class 
loc:@Predicates/dep/Variable*
T0
?
$Predicates/dep/embedding_lookup/axisConst*
value	B : *
_output_shapes
: **
_class 
loc:@Predicates/dep/Variable*
dtype0
?
Predicates/dep/embedding_lookupGatherV2Predicates/dep/Variable/readpred_dep$Predicates/dep/embedding_lookup/axis*
Tparams0*
Taxis0*
Tindices0*4
_output_shapes"
 :??????????????????**
_class 
loc:@Predicates/dep/Variable
?
(Predicates/dep/embedding_lookup/IdentityIdentityPredicates/dep/embedding_lookup*4
_output_shapes"
 :??????????????????*
T0
u
$Predicates/mask/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
g
"Predicates/mask/random_uniform/minConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
g
"Predicates/mask/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
,Predicates/mask/random_uniform/RandomUniformRandomUniform$Predicates/mask/random_uniform/shape*
_output_shapes

:*
T0*
dtype0
?
"Predicates/mask/random_uniform/subSub"Predicates/mask/random_uniform/max"Predicates/mask/random_uniform/min*
T0*
_output_shapes
: 
?
"Predicates/mask/random_uniform/mulMul,Predicates/mask/random_uniform/RandomUniform"Predicates/mask/random_uniform/sub*
_output_shapes

:*
T0
?
Predicates/mask/random_uniformAdd"Predicates/mask/random_uniform/mul"Predicates/mask/random_uniform/min*
T0*
_output_shapes

:
h
Predicates/mask/Variable
VariableV2*
dtype0*
shape
:*
_output_shapes

:
?
Predicates/mask/Variable/AssignAssignPredicates/mask/VariablePredicates/mask/random_uniform*+
_class!
loc:@Predicates/mask/Variable*
T0*
_output_shapes

:
?
Predicates/mask/Variable/readIdentityPredicates/mask/Variable*+
_class!
loc:@Predicates/mask/Variable*
_output_shapes

:*
T0
?
%Predicates/mask/embedding_lookup/axisConst*+
_class!
loc:@Predicates/mask/Variable*
value	B : *
_output_shapes
: *
dtype0
?
 Predicates/mask/embedding_lookupGatherV2Predicates/mask/Variable/read	pred_mask%Predicates/mask/embedding_lookup/axis*+
_class!
loc:@Predicates/mask/Variable*
Tparams0*
Tindices0*4
_output_shapes"
 :??????????????????*
Taxis0
?
)Predicates/mask/embedding_lookup/IdentityIdentity Predicates/mask/embedding_lookup*4
_output_shapes"
 :??????????????????*
T0
a
Predicates/concat/axisConst*
valueB :
?????????*
_output_shapes
: *
dtype0
?
Predicates/concatConcatV2(Predicates/pos/embedding_lookup/Identity(Predicates/dep/embedding_lookup/IdentityPredicates/concat/axis*
N*4
_output_shapes"
 :??????????????????(*
T0
c
Predicates/concat_1/axisConst*
valueB :
?????????*
_output_shapes
: *
dtype0
?
Predicates/concat_1ConcatV2Predicates/concat)Predicates/mask/embedding_lookup/IdentityPredicates/concat_1/axis*
N*4
_output_shapes"
 :??????????????????<*
T0
?
%Predicates/index_to_string/asset_pathConst"/device:CPU:**
valueB Bindex_tag.txt*
_output_shapes
: *
dtype0
d
 Predicates/index_to_string/ConstConst*
valueB	 BUNK*
_output_shapes
: *
dtype0
?
%Predicates/index_to_string/hash_tableHashTableV2*
_output_shapes
: *
value_dtype0*/
shared_name hash_table_index_tag.txt_-1_-2*
	key_dtype0	
?
CPredicates/index_to_string/table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2%Predicates/index_to_string/hash_table%Predicates/index_to_string/asset_path*
value_index?????????*
	key_index?????????
w
2Predicates/BiLSTM/forward/DropoutWrapperInit/ConstConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
y
4Predicates/BiLSTM/forward/DropoutWrapperInit/Const_1Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
y
4Predicates/BiLSTM/forward/DropoutWrapperInit/Const_2Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
s
)Predicates/BiLSTM/forward/concat/values_0Const*
valueB:*
_output_shapes
:*
dtype0
t
)Predicates/BiLSTM/forward/concat/values_1Const*
valueB:?*
_output_shapes
:*
dtype0
g
%Predicates/BiLSTM/forward/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
 Predicates/BiLSTM/forward/concatConcatV2)Predicates/BiLSTM/forward/concat/values_0)Predicates/BiLSTM/forward/concat/values_1%Predicates/BiLSTM/forward/concat/axis*
N*
_output_shapes
:*
T0
q
,Predicates/BiLSTM/forward/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
q
,Predicates/BiLSTM/forward/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
6Predicates/BiLSTM/forward/random_uniform/RandomUniformRandomUniform Predicates/BiLSTM/forward/concat*
_output_shapes
:	?*
T0*
dtype0
?
,Predicates/BiLSTM/forward/random_uniform/subSub,Predicates/BiLSTM/forward/random_uniform/max,Predicates/BiLSTM/forward/random_uniform/min*
_output_shapes
: *
T0
?
,Predicates/BiLSTM/forward/random_uniform/mulMul6Predicates/BiLSTM/forward/random_uniform/RandomUniform,Predicates/BiLSTM/forward/random_uniform/sub*
_output_shapes
:	?*
T0
?
(Predicates/BiLSTM/forward/random_uniformAdd,Predicates/BiLSTM/forward/random_uniform/mul,Predicates/BiLSTM/forward/random_uniform/min*
_output_shapes
:	?*
T0
u
+Predicates/BiLSTM/forward/concat_1/values_0Const*
valueB:*
_output_shapes
:*
dtype0
v
+Predicates/BiLSTM/forward/concat_1/values_1Const*
valueB:?*
_output_shapes
:*
dtype0
i
'Predicates/BiLSTM/forward/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
"Predicates/BiLSTM/forward/concat_1ConcatV2+Predicates/BiLSTM/forward/concat_1/values_0+Predicates/BiLSTM/forward/concat_1/values_1'Predicates/BiLSTM/forward/concat_1/axis*
_output_shapes
:*
T0*
N
s
.Predicates/BiLSTM/forward/random_uniform_1/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
s
.Predicates/BiLSTM/forward/random_uniform_1/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
8Predicates/BiLSTM/forward/random_uniform_1/RandomUniformRandomUniform"Predicates/BiLSTM/forward/concat_1*
_output_shapes
:	?*
dtype0*
T0
?
.Predicates/BiLSTM/forward/random_uniform_1/subSub.Predicates/BiLSTM/forward/random_uniform_1/max.Predicates/BiLSTM/forward/random_uniform_1/min*
_output_shapes
: *
T0
?
.Predicates/BiLSTM/forward/random_uniform_1/mulMul8Predicates/BiLSTM/forward/random_uniform_1/RandomUniform.Predicates/BiLSTM/forward/random_uniform_1/sub*
_output_shapes
:	?*
T0
?
*Predicates/BiLSTM/forward/random_uniform_1Add.Predicates/BiLSTM/forward/random_uniform_1/mul.Predicates/BiLSTM/forward/random_uniform_1/min*
_output_shapes
:	?*
T0
u
+Predicates/BiLSTM/forward/concat_2/values_0Const*
valueB:*
_output_shapes
:*
dtype0
v
+Predicates/BiLSTM/forward/concat_2/values_1Const*
valueB:?*
_output_shapes
:*
dtype0
i
'Predicates/BiLSTM/forward/concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
"Predicates/BiLSTM/forward/concat_2ConcatV2+Predicates/BiLSTM/forward/concat_2/values_0+Predicates/BiLSTM/forward/concat_2/values_1'Predicates/BiLSTM/forward/concat_2/axis*
_output_shapes
:*
N*
T0
s
.Predicates/BiLSTM/forward/random_uniform_2/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
s
.Predicates/BiLSTM/forward/random_uniform_2/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
8Predicates/BiLSTM/forward/random_uniform_2/RandomUniformRandomUniform"Predicates/BiLSTM/forward/concat_2*
dtype0*
T0*
_output_shapes
:	?
?
.Predicates/BiLSTM/forward/random_uniform_2/subSub.Predicates/BiLSTM/forward/random_uniform_2/max.Predicates/BiLSTM/forward/random_uniform_2/min*
_output_shapes
: *
T0
?
.Predicates/BiLSTM/forward/random_uniform_2/mulMul8Predicates/BiLSTM/forward/random_uniform_2/RandomUniform.Predicates/BiLSTM/forward/random_uniform_2/sub*
_output_shapes
:	?*
T0
?
*Predicates/BiLSTM/forward/random_uniform_2Add.Predicates/BiLSTM/forward/random_uniform_2/mul.Predicates/BiLSTM/forward/random_uniform_2/min*
_output_shapes
:	?*
T0
x
3Predicates/BiLSTM/backward/DropoutWrapperInit/ConstConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
z
5Predicates/BiLSTM/backward/DropoutWrapperInit/Const_1Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
z
5Predicates/BiLSTM/backward/DropoutWrapperInit/Const_2Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
t
*Predicates/BiLSTM/backward/concat/values_0Const*
valueB:*
_output_shapes
:*
dtype0
u
*Predicates/BiLSTM/backward/concat/values_1Const*
valueB:?*
_output_shapes
:*
dtype0
h
&Predicates/BiLSTM/backward/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
!Predicates/BiLSTM/backward/concatConcatV2*Predicates/BiLSTM/backward/concat/values_0*Predicates/BiLSTM/backward/concat/values_1&Predicates/BiLSTM/backward/concat/axis*
T0*
_output_shapes
:*
N
r
-Predicates/BiLSTM/backward/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-Predicates/BiLSTM/backward/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
7Predicates/BiLSTM/backward/random_uniform/RandomUniformRandomUniform!Predicates/BiLSTM/backward/concat*
dtype0*
T0*
_output_shapes
:	?
?
-Predicates/BiLSTM/backward/random_uniform/subSub-Predicates/BiLSTM/backward/random_uniform/max-Predicates/BiLSTM/backward/random_uniform/min*
T0*
_output_shapes
: 
?
-Predicates/BiLSTM/backward/random_uniform/mulMul7Predicates/BiLSTM/backward/random_uniform/RandomUniform-Predicates/BiLSTM/backward/random_uniform/sub*
T0*
_output_shapes
:	?
?
)Predicates/BiLSTM/backward/random_uniformAdd-Predicates/BiLSTM/backward/random_uniform/mul-Predicates/BiLSTM/backward/random_uniform/min*
T0*
_output_shapes
:	?
v
,Predicates/BiLSTM/backward/concat_1/values_0Const*
dtype0*
valueB:*
_output_shapes
:
w
,Predicates/BiLSTM/backward/concat_1/values_1Const*
dtype0*
valueB:?*
_output_shapes
:
j
(Predicates/BiLSTM/backward/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
#Predicates/BiLSTM/backward/concat_1ConcatV2,Predicates/BiLSTM/backward/concat_1/values_0,Predicates/BiLSTM/backward/concat_1/values_1(Predicates/BiLSTM/backward/concat_1/axis*
_output_shapes
:*
T0*
N
t
/Predicates/BiLSTM/backward/random_uniform_1/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
t
/Predicates/BiLSTM/backward/random_uniform_1/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
9Predicates/BiLSTM/backward/random_uniform_1/RandomUniformRandomUniform#Predicates/BiLSTM/backward/concat_1*
_output_shapes
:	?*
dtype0*
T0
?
/Predicates/BiLSTM/backward/random_uniform_1/subSub/Predicates/BiLSTM/backward/random_uniform_1/max/Predicates/BiLSTM/backward/random_uniform_1/min*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/backward/random_uniform_1/mulMul9Predicates/BiLSTM/backward/random_uniform_1/RandomUniform/Predicates/BiLSTM/backward/random_uniform_1/sub*
_output_shapes
:	?*
T0
?
+Predicates/BiLSTM/backward/random_uniform_1Add/Predicates/BiLSTM/backward/random_uniform_1/mul/Predicates/BiLSTM/backward/random_uniform_1/min*
_output_shapes
:	?*
T0
v
,Predicates/BiLSTM/backward/concat_2/values_0Const*
valueB:*
_output_shapes
:*
dtype0
w
,Predicates/BiLSTM/backward/concat_2/values_1Const*
valueB:?*
_output_shapes
:*
dtype0
j
(Predicates/BiLSTM/backward/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
#Predicates/BiLSTM/backward/concat_2ConcatV2,Predicates/BiLSTM/backward/concat_2/values_0,Predicates/BiLSTM/backward/concat_2/values_1(Predicates/BiLSTM/backward/concat_2/axis*
_output_shapes
:*
N*
T0
t
/Predicates/BiLSTM/backward/random_uniform_2/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
t
/Predicates/BiLSTM/backward/random_uniform_2/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
9Predicates/BiLSTM/backward/random_uniform_2/RandomUniformRandomUniform#Predicates/BiLSTM/backward/concat_2*
_output_shapes
:	?*
dtype0*
T0
?
/Predicates/BiLSTM/backward/random_uniform_2/subSub/Predicates/BiLSTM/backward/random_uniform_2/max/Predicates/BiLSTM/backward/random_uniform_2/min*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/backward/random_uniform_2/mulMul9Predicates/BiLSTM/backward/random_uniform_2/RandomUniform/Predicates/BiLSTM/backward/random_uniform_2/sub*
_output_shapes
:	?*
T0
?
+Predicates/BiLSTM/backward/random_uniform_2Add/Predicates/BiLSTM/backward/random_uniform_2/mul/Predicates/BiLSTM/backward/random_uniform_2/min*
_output_shapes
:	?*
T0
e
#Predicates/BiLSTM/BiLSTM/fw/fw/RankConst*
_output_shapes
: *
dtype0*
value	B :
l
*Predicates/BiLSTM/BiLSTM/fw/fw/range/startConst*
_output_shapes
: *
dtype0*
value	B :
l
*Predicates/BiLSTM/BiLSTM/fw/fw/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
$Predicates/BiLSTM/BiLSTM/fw/fw/rangeRange*Predicates/BiLSTM/BiLSTM/fw/fw/range/start#Predicates/BiLSTM/BiLSTM/fw/fw/Rank*Predicates/BiLSTM/BiLSTM/fw/fw/range/delta*
_output_shapes
:

.Predicates/BiLSTM/BiLSTM/fw/fw/concat/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
l
*Predicates/BiLSTM/BiLSTM/fw/fw/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
%Predicates/BiLSTM/BiLSTM/fw/fw/concatConcatV2.Predicates/BiLSTM/BiLSTM/fw/fw/concat/values_0$Predicates/BiLSTM/BiLSTM/fw/fw/range*Predicates/BiLSTM/BiLSTM/fw/fw/concat/axis*
N*
_output_shapes
:*
T0
?
(Predicates/BiLSTM/BiLSTM/fw/fw/transpose	TransposePredicates/concat_1%Predicates/BiLSTM/BiLSTM/fw/fw/concat*4
_output_shapes"
 :??????????????????<*
T0
?
.Predicates/BiLSTM/BiLSTM/fw/fw/sequence_lengthIdentitypred_original_sequence_lengths*#
_output_shapes
:?????????*
T0
|
$Predicates/BiLSTM/BiLSTM/fw/fw/ShapeShape(Predicates/BiLSTM/BiLSTM/fw/fw/transpose*
_output_shapes
:*
T0
|
2Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
~
4Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
~
4Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
,Predicates/BiLSTM/BiLSTM/fw/fw/strided_sliceStridedSlice$Predicates/BiLSTM/BiLSTM/fw/fw/Shape2Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice/stack4Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_14Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
?
\Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
XPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims
ExpandDims,Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice\Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:
?
SPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstConst*
dtype0*
valueB:?*
_output_shapes
:
?
YPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
TPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatConcatV2XPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDimsSPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstYPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axis*
T0*
_output_shapes
:*
N
?
YPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
SPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zerosFillTPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatYPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/Const*
T0*(
_output_shapes
:??????????
?
^Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
ZPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1
ExpandDims,Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice^Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:
?
UPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_1Const*
dtype0*
valueB:?*
_output_shapes
:
?
^Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
ZPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2
ExpandDims,Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice^Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dim*
T0*
_output_shapes
:
?
UPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2Const*
dtype0*
valueB:?*
_output_shapes
:
?
[Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
VPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1ConcatV2ZPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2UPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2[Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:
?
[Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
UPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1FillVPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1[Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/Const*
T0*(
_output_shapes
:??????????
?
^Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
ZPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3
ExpandDims,Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice^Predicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:
?
UPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_3Const*
dtype0*
valueB:?*
_output_shapes
:
?
&Predicates/BiLSTM/BiLSTM/fw/fw/Shape_1Shape.Predicates/BiLSTM/BiLSTM/fw/fw/sequence_length*
T0*
_output_shapes
:
?
$Predicates/BiLSTM/BiLSTM/fw/fw/stackPack,Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice*
T0*
N*
_output_shapes
:
?
$Predicates/BiLSTM/BiLSTM/fw/fw/EqualEqual&Predicates/BiLSTM/BiLSTM/fw/fw/Shape_1$Predicates/BiLSTM/BiLSTM/fw/fw/stack*
T0*
_output_shapes
:
n
$Predicates/BiLSTM/BiLSTM/fw/fw/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
"Predicates/BiLSTM/BiLSTM/fw/fw/AllAll$Predicates/BiLSTM/BiLSTM/fw/fw/Equal$Predicates/BiLSTM/BiLSTM/fw/fw/Const*
_output_shapes
: 
?
+Predicates/BiLSTM/BiLSTM/fw/fw/Assert/ConstConst*
dtype0*_
valueVBT BNExpected shape for Tensor Predicates/BiLSTM/BiLSTM/fw/fw/sequence_length:0 is *
_output_shapes
: 
~
-Predicates/BiLSTM/BiLSTM/fw/fw/Assert/Const_1Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
?
3Predicates/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_0Const*
dtype0*_
valueVBT BNExpected shape for Tensor Predicates/BiLSTM/BiLSTM/fw/fw/sequence_length:0 is *
_output_shapes
: 
?
3Predicates/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_2Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
?
,Predicates/BiLSTM/BiLSTM/fw/fw/Assert/AssertAssert"Predicates/BiLSTM/BiLSTM/fw/fw/All3Predicates/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_0$Predicates/BiLSTM/BiLSTM/fw/fw/stack3Predicates/BiLSTM/BiLSTM/fw/fw/Assert/Assert/data_2&Predicates/BiLSTM/BiLSTM/fw/fw/Shape_1*
T
2
?
*Predicates/BiLSTM/BiLSTM/fw/fw/CheckSeqLenIdentity.Predicates/BiLSTM/BiLSTM/fw/fw/sequence_length-^Predicates/BiLSTM/BiLSTM/fw/fw/Assert/Assert*
T0*#
_output_shapes
:?????????
~
&Predicates/BiLSTM/BiLSTM/fw/fw/Shape_2Shape(Predicates/BiLSTM/BiLSTM/fw/fw/transpose*
T0*
_output_shapes
:
~
4Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
?
6Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
?
6Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
.Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1StridedSlice&Predicates/BiLSTM/BiLSTM/fw/fw/Shape_24Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack6Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_16Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
~
&Predicates/BiLSTM/BiLSTM/fw/fw/Shape_3Shape(Predicates/BiLSTM/BiLSTM/fw/fw/transpose*
_output_shapes
:*
T0
~
4Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
6Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
6Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
.Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2StridedSlice&Predicates/BiLSTM/BiLSTM/fw/fw/Shape_34Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack6Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_16Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
o
-Predicates/BiLSTM/BiLSTM/fw/fw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
)Predicates/BiLSTM/BiLSTM/fw/fw/ExpandDims
ExpandDims.Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_2-Predicates/BiLSTM/BiLSTM/fw/fw/ExpandDims/dim*
_output_shapes
:*
T0
q
&Predicates/BiLSTM/BiLSTM/fw/fw/Const_1Const*
_output_shapes
:*
dtype0*
valueB:?
n
,Predicates/BiLSTM/BiLSTM/fw/fw/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
'Predicates/BiLSTM/BiLSTM/fw/fw/concat_1ConcatV2)Predicates/BiLSTM/BiLSTM/fw/fw/ExpandDims&Predicates/BiLSTM/BiLSTM/fw/fw/Const_1,Predicates/BiLSTM/BiLSTM/fw/fw/concat_1/axis*
T0*
_output_shapes
:*
N
o
*Predicates/BiLSTM/BiLSTM/fw/fw/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
$Predicates/BiLSTM/BiLSTM/fw/fw/zerosFill'Predicates/BiLSTM/BiLSTM/fw/fw/concat_1*Predicates/BiLSTM/BiLSTM/fw/fw/zeros/Const*
T0*(
_output_shapes
:??????????
p
&Predicates/BiLSTM/BiLSTM/fw/fw/Const_2Const*
valueB: *
_output_shapes
:*
dtype0
?
"Predicates/BiLSTM/BiLSTM/fw/fw/MinMin*Predicates/BiLSTM/BiLSTM/fw/fw/CheckSeqLen&Predicates/BiLSTM/BiLSTM/fw/fw/Const_2*
T0*
_output_shapes
: 
p
&Predicates/BiLSTM/BiLSTM/fw/fw/Const_3Const*
valueB: *
_output_shapes
:*
dtype0
?
"Predicates/BiLSTM/BiLSTM/fw/fw/MaxMax*Predicates/BiLSTM/BiLSTM/fw/fw/CheckSeqLen&Predicates/BiLSTM/BiLSTM/fw/fw/Const_3*
_output_shapes
: *
T0
e
#Predicates/BiLSTM/BiLSTM/fw/fw/timeConst*
value	B : *
_output_shapes
: *
dtype0
?
*Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayTensorArrayV3.Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1*
_output_shapes

:: *
identical_element_shapes(*J
tensor_array_name53Predicates/BiLSTM/BiLSTM/fw/fw/dynamic_rnn/output_0*%
element_shape:??????????*
dtype0
?
,Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray_1TensorArrayV3.Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1*
identical_element_shapes(*
_output_shapes

:: *$
element_shape:?????????<*
dtype0*I
tensor_array_name42Predicates/BiLSTM/BiLSTM/fw/fw/dynamic_rnn/input_0
?
7Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/ShapeShape(Predicates/BiLSTM/BiLSTM/fw/fw/transpose*
_output_shapes
:*
T0
?
EPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
?
GPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
?
GPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
?Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice7Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/ShapeEPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stackGPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_1GPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
shrink_axis_mask*
_output_shapes
: *
T0

=Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0

=Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
7Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/rangeRange=Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/start?Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/strided_slice=Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????
?
YPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3,Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray_17Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/range(Predicates/BiLSTM/BiLSTM/fw/fw/transpose.Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray_1:1*
_output_shapes
: *
T0*;
_class1
/-loc:@Predicates/BiLSTM/BiLSTM/fw/fw/transpose
j
(Predicates/BiLSTM/BiLSTM/fw/fw/Maximum/xConst*
value	B :*
_output_shapes
: *
dtype0
?
&Predicates/BiLSTM/BiLSTM/fw/fw/MaximumMaximum(Predicates/BiLSTM/BiLSTM/fw/fw/Maximum/x"Predicates/BiLSTM/BiLSTM/fw/fw/Max*
T0*
_output_shapes
: 
?
&Predicates/BiLSTM/BiLSTM/fw/fw/MinimumMinimum.Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1&Predicates/BiLSTM/BiLSTM/fw/fw/Maximum*
_output_shapes
: *
T0
x
6Predicates/BiLSTM/BiLSTM/fw/fw/while/iteration_counterConst*
dtype0*
value	B : *
_output_shapes
: 
?
*Predicates/BiLSTM/BiLSTM/fw/fw/while/EnterEnter6Predicates/BiLSTM/BiLSTM/fw/fw/while/iteration_counter*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_1Enter#Predicates/BiLSTM/BiLSTM/fw/fw/time*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_2Enter,Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray:1*
parallel_iterations *
T0*
_output_shapes
: *B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_3EnterSPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros*
parallel_iterations *
T0*(
_output_shapes
:??????????*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_4EnterUPredicates/BiLSTM/BiLSTM/fw/fw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1*(
_output_shapes
:??????????*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
parallel_iterations *
T0
?
*Predicates/BiLSTM/BiLSTM/fw/fw/while/MergeMerge*Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter2Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration*
T0*
_output_shapes
: : *
N
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_1Merge,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_14Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_1*
T0*
_output_shapes
: : *
N
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_2Merge,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_24Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_2*
N*
_output_shapes
: : *
T0
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_3Merge,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_34Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_3*
N**
_output_shapes
:??????????: *
T0
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_4Merge,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_44Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_4*
N**
_output_shapes
:??????????: *
T0
?
)Predicates/BiLSTM/BiLSTM/fw/fw/while/LessLess*Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge/Predicates/BiLSTM/BiLSTM/fw/fw/while/Less/Enter*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Less/EnterEnter.Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1*
is_constant(*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
?
+Predicates/BiLSTM/BiLSTM/fw/fw/while/Less_1Less,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_11Predicates/BiLSTM/BiLSTM/fw/fw/while/Less_1/Enter*
_output_shapes
: *
T0
?
1Predicates/BiLSTM/BiLSTM/fw/fw/while/Less_1/EnterEnter&Predicates/BiLSTM/BiLSTM/fw/fw/Minimum*
is_constant(*
_output_shapes
: *
parallel_iterations *B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0
?
/Predicates/BiLSTM/BiLSTM/fw/fw/while/LogicalAnd
LogicalAnd)Predicates/BiLSTM/BiLSTM/fw/fw/while/Less+Predicates/BiLSTM/BiLSTM/fw/fw/while/Less_1*
_output_shapes
: 
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCondLoopCond/Predicates/BiLSTM/BiLSTM/fw/fw/while/LogicalAnd*
_output_shapes
: 
?
+Predicates/BiLSTM/BiLSTM/fw/fw/while/SwitchSwitch*Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge-Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCond*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge*
_output_shapes
: : *
T0
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_1Switch,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_1-Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCond*
_output_shapes
: : *
T0*?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_1
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_2Switch,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_2-Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCond*
_output_shapes
: : *
T0*?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_2
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_3Switch,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_3-Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCond*<
_output_shapes*
(:??????????:??????????*
T0*?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_3
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_4Switch,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_4-Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCond*<
_output_shapes*
(:??????????:??????????*
T0*?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_4
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/IdentityIdentity-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch:1*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_1Identity/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_1:1*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_2Identity/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_2:1*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_3Identity/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_3:1*(
_output_shapes
:??????????*
T0
?
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_4Identity/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_4:1*(
_output_shapes
:??????????*
T0
?
*Predicates/BiLSTM/BiLSTM/fw/fw/while/add/yConst.^Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
(Predicates/BiLSTM/BiLSTM/fw/fw/while/addAdd-Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity*Predicates/BiLSTM/BiLSTM/fw/fw/while/add/y*
_output_shapes
: *
T0
?
6Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3TensorArrayReadV3<Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_1>Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:?????????<*
dtype0
?
<Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/EnterEnter,Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray_1*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
is_constant(*
_output_shapes
:*
parallel_iterations *
T0
?
>Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1EnterYPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
is_constant(*
_output_shapes
: *
T0*
parallel_iterations 
?
1Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqualGreaterEqual/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_17Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Enter*#
_output_shapes
:?????????*
T0
?
7Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/EnterEnter*Predicates/BiLSTM/BiLSTM/fw/fw/CheckSeqLen*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
is_constant(*#
_output_shapes
:?????????*
T0*
parallel_iterations 
?
FPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"?      *
_output_shapes
:*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
dtype0
?
DPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/minConst*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
valueB
 *????*
dtype0*
_output_shapes
: 
?
DPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *???=*
_output_shapes
: *8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
dtype0
?
NPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformFPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
dtype0*
T0
?
DPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/subSubDPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/maxDPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
T0
?
DPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/mulMulNPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformDPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
T0
?
@Predicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniformAddDPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/mulDPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
T0
?
%Predicates/BiLSTM/fw/lstm_cell/kernel
VariableV2* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
shape:
??*
dtype0
?
,Predicates/BiLSTM/fw/lstm_cell/kernel/AssignAssign%Predicates/BiLSTM/fw/lstm_cell/kernel@Predicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel*
T0
?
*Predicates/BiLSTM/fw/lstm_cell/kernel/readIdentity%Predicates/BiLSTM/fw/lstm_cell/kernel* 
_output_shapes
:
??*
T0
?
5Predicates/BiLSTM/fw/lstm_cell/bias/Initializer/ConstConst*
valueB?*    *
_output_shapes	
:?*6
_class,
*(loc:@Predicates/BiLSTM/fw/lstm_cell/bias*
dtype0
?
#Predicates/BiLSTM/fw/lstm_cell/bias
VariableV2*
_output_shapes	
:?*
dtype0*6
_class,
*(loc:@Predicates/BiLSTM/fw/lstm_cell/bias*
shape:?
?
*Predicates/BiLSTM/fw/lstm_cell/bias/AssignAssign#Predicates/BiLSTM/fw/lstm_cell/bias5Predicates/BiLSTM/fw/lstm_cell/bias/Initializer/Const*
_output_shapes	
:?*6
_class,
*(loc:@Predicates/BiLSTM/fw/lstm_cell/bias*
T0

(Predicates/BiLSTM/fw/lstm_cell/bias/readIdentity#Predicates/BiLSTM/fw/lstm_cell/bias*
_output_shapes	
:?*
T0
?
4Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zerosConst.^Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity*
valueB?*    *
dtype0*
_output_shapes	
:?
?
<Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCellLSTMBlockCell6Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_3/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_4BPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter4Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zeros4Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zeros4Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zerosDPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????*
T0*
	cell_clip%  ??
?
BPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/EnterEnter*Predicates/BiLSTM/fw/lstm_cell/kernel/read*
parallel_iterations * 
_output_shapes
:
??*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0*
is_constant(
?
DPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1Enter(Predicates/BiLSTM/fw/lstm_cell/bias/read*
parallel_iterations *
_output_shapes	
:?*B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*
T0*
is_constant(
?
+Predicates/BiLSTM/BiLSTM/fw/fw/while/SelectSelect1Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual1Predicates/BiLSTM/BiLSTM/fw/fw/while/Select/Enter>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:6*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
T0*(
_output_shapes
:??????????
?
1Predicates/BiLSTM/BiLSTM/fw/fw/while/Select/EnterEnter$Predicates/BiLSTM/BiLSTM/fw/fw/zeros*
parallel_iterations *B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
is_constant(*
T0*(
_output_shapes
:??????????
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Select_1Select1Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_3>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:1*(
_output_shapes
:??????????*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
T0
?
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Select_2Select1Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_4>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:6*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
T0*(
_output_shapes
:??????????
?
HPredicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3NPredicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_1+Predicates/BiLSTM/BiLSTM/fw/fw/while/Select/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_2*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
T0*
_output_shapes
: 
?
NPredicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter*Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray*
parallel_iterations *B

frame_name42Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell*
is_constant(*
T0*
_output_shapes
:
?
,Predicates/BiLSTM/BiLSTM/fw/fw/while/add_1/yConst.^Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity*
dtype0*
value	B :*
_output_shapes
: 
?
*Predicates/BiLSTM/BiLSTM/fw/fw/while/add_1Add/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_1,Predicates/BiLSTM/BiLSTM/fw/fw/while/add_1/y*
T0*
_output_shapes
: 
?
2Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIterationNextIteration(Predicates/BiLSTM/BiLSTM/fw/fw/while/add*
_output_shapes
: *
T0
?
4Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_1NextIteration*Predicates/BiLSTM/BiLSTM/fw/fw/while/add_1*
_output_shapes
: *
T0
?
4Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_2NextIterationHPredicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
?
4Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_3NextIteration-Predicates/BiLSTM/BiLSTM/fw/fw/while/Select_1*(
_output_shapes
:??????????*
T0
?
4Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_4NextIteration-Predicates/BiLSTM/BiLSTM/fw/fw/while/Select_2*(
_output_shapes
:??????????*
T0

)Predicates/BiLSTM/BiLSTM/fw/fw/while/ExitExit+Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch*
_output_shapes
: *
T0
?
+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_1Exit-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_1*
_output_shapes
: *
T0
?
+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_2Exit-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_2*
_output_shapes
: *
T0
?
+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_3Exit-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_3*(
_output_shapes
:??????????*
T0
?
+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_4Exit-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_4*(
_output_shapes
:??????????*
T0
?
APredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3*Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_2*
_output_shapes
: *=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray
?
;Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/startConst*
value	B : *
dtype0*
_output_shapes
: *=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray
?
;Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/deltaConst*
_output_shapes
: *
dtype0*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray*
value	B :
?
5Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/rangeRange;Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/startAPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArraySizeV3;Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range/delta*#
_output_shapes
:?????????*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray
?
CPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3*Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray5Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/range+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_2*5
_output_shapes#
!:???????????????????*%
element_shape:??????????*
dtype0*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray
q
&Predicates/BiLSTM/BiLSTM/fw/fw/Const_4Const*
valueB:?*
_output_shapes
:*
dtype0
g
%Predicates/BiLSTM/BiLSTM/fw/fw/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
n
,Predicates/BiLSTM/BiLSTM/fw/fw/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :
n
,Predicates/BiLSTM/BiLSTM/fw/fw/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
&Predicates/BiLSTM/BiLSTM/fw/fw/range_1Range,Predicates/BiLSTM/BiLSTM/fw/fw/range_1/start%Predicates/BiLSTM/BiLSTM/fw/fw/Rank_1,Predicates/BiLSTM/BiLSTM/fw/fw/range_1/delta*
_output_shapes
:
?
0Predicates/BiLSTM/BiLSTM/fw/fw/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
n
,Predicates/BiLSTM/BiLSTM/fw/fw/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
'Predicates/BiLSTM/BiLSTM/fw/fw/concat_2ConcatV20Predicates/BiLSTM/BiLSTM/fw/fw/concat_2/values_0&Predicates/BiLSTM/BiLSTM/fw/fw/range_1,Predicates/BiLSTM/BiLSTM/fw/fw/concat_2/axis*
N*
_output_shapes
:*
T0
?
*Predicates/BiLSTM/BiLSTM/fw/fw/transpose_1	TransposeCPredicates/BiLSTM/BiLSTM/fw/fw/TensorArrayStack/TensorArrayGatherV3'Predicates/BiLSTM/BiLSTM/fw/fw/concat_2*5
_output_shapes#
!:???????????????????*
T0
?
+Predicates/BiLSTM/BiLSTM/bw/ReverseSequenceReverseSequencePredicates/concat_1pred_original_sequence_lengths*4
_output_shapes"
 :??????????????????<*
T0*
seq_dim*

Tlen0
e
#Predicates/BiLSTM/BiLSTM/bw/bw/RankConst*
value	B :*
_output_shapes
: *
dtype0
l
*Predicates/BiLSTM/BiLSTM/bw/bw/range/startConst*
value	B :*
_output_shapes
: *
dtype0
l
*Predicates/BiLSTM/BiLSTM/bw/bw/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
$Predicates/BiLSTM/BiLSTM/bw/bw/rangeRange*Predicates/BiLSTM/BiLSTM/bw/bw/range/start#Predicates/BiLSTM/BiLSTM/bw/bw/Rank*Predicates/BiLSTM/BiLSTM/bw/bw/range/delta*
_output_shapes
:

.Predicates/BiLSTM/BiLSTM/bw/bw/concat/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
l
*Predicates/BiLSTM/BiLSTM/bw/bw/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
%Predicates/BiLSTM/BiLSTM/bw/bw/concatConcatV2.Predicates/BiLSTM/BiLSTM/bw/bw/concat/values_0$Predicates/BiLSTM/BiLSTM/bw/bw/range*Predicates/BiLSTM/BiLSTM/bw/bw/concat/axis*
T0*
_output_shapes
:*
N
?
(Predicates/BiLSTM/BiLSTM/bw/bw/transpose	Transpose+Predicates/BiLSTM/BiLSTM/bw/ReverseSequence%Predicates/BiLSTM/BiLSTM/bw/bw/concat*
T0*4
_output_shapes"
 :??????????????????<
?
.Predicates/BiLSTM/BiLSTM/bw/bw/sequence_lengthIdentitypred_original_sequence_lengths*
T0*#
_output_shapes
:?????????
|
$Predicates/BiLSTM/BiLSTM/bw/bw/ShapeShape(Predicates/BiLSTM/BiLSTM/bw/bw/transpose*
T0*
_output_shapes
:
|
2Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
~
4Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
~
4Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
,Predicates/BiLSTM/BiLSTM/bw/bw/strided_sliceStridedSlice$Predicates/BiLSTM/BiLSTM/bw/bw/Shape2Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice/stack4Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_14Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
?
\Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
XPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims
ExpandDims,Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice\Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:
?
SPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstConst*
_output_shapes
:*
dtype0*
valueB:?
?
YPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
TPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatConcatV2XPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDimsSPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstYPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axis*
_output_shapes
:*
N*
T0
?
YPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
?
SPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zerosFillTPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatYPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/Const*(
_output_shapes
:??????????*
T0
?
^Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dimConst*
_output_shapes
: *
value	B : *
dtype0
?
ZPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1
ExpandDims,Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice^Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_1/dim*
_output_shapes
:*
T0
?
UPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_1Const*
_output_shapes
:*
valueB:?*
dtype0
?
^Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dimConst*
_output_shapes
: *
value	B : *
dtype0
?
ZPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2
ExpandDims,Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice^Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2/dim*
_output_shapes
:*
T0
?
UPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2Const*
_output_shapes
:*
valueB:?*
dtype0
?
[Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
VPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1ConcatV2ZPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_2UPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2[Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axis*
_output_shapes
:*
N*
T0
?
[Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
?
UPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1FillVPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1[Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/Const*(
_output_shapes
:??????????*
T0
?
^Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
ZPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3
ExpandDims,Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice^Predicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:
?
UPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_3Const*
_output_shapes
:*
dtype0*
valueB:?
?
&Predicates/BiLSTM/BiLSTM/bw/bw/Shape_1Shape.Predicates/BiLSTM/BiLSTM/bw/bw/sequence_length*
_output_shapes
:*
T0
?
$Predicates/BiLSTM/BiLSTM/bw/bw/stackPack,Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice*
_output_shapes
:*
N*
T0
?
$Predicates/BiLSTM/BiLSTM/bw/bw/EqualEqual&Predicates/BiLSTM/BiLSTM/bw/bw/Shape_1$Predicates/BiLSTM/BiLSTM/bw/bw/stack*
_output_shapes
:*
T0
n
$Predicates/BiLSTM/BiLSTM/bw/bw/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
"Predicates/BiLSTM/BiLSTM/bw/bw/AllAll$Predicates/BiLSTM/BiLSTM/bw/bw/Equal$Predicates/BiLSTM/BiLSTM/bw/bw/Const*
_output_shapes
: 
?
+Predicates/BiLSTM/BiLSTM/bw/bw/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNExpected shape for Tensor Predicates/BiLSTM/BiLSTM/bw/bw/sequence_length:0 is 
~
-Predicates/BiLSTM/BiLSTM/bw/bw/Assert/Const_1Const*
_output_shapes
: *
dtype0*!
valueB B but saw shape: 
?
3Predicates/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNExpected shape for Tensor Predicates/BiLSTM/BiLSTM/bw/bw/sequence_length:0 is 
?
3Predicates/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*!
valueB B but saw shape: 
?
,Predicates/BiLSTM/BiLSTM/bw/bw/Assert/AssertAssert"Predicates/BiLSTM/BiLSTM/bw/bw/All3Predicates/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_0$Predicates/BiLSTM/BiLSTM/bw/bw/stack3Predicates/BiLSTM/BiLSTM/bw/bw/Assert/Assert/data_2&Predicates/BiLSTM/BiLSTM/bw/bw/Shape_1*
T
2
?
*Predicates/BiLSTM/BiLSTM/bw/bw/CheckSeqLenIdentity.Predicates/BiLSTM/BiLSTM/bw/bw/sequence_length-^Predicates/BiLSTM/BiLSTM/bw/bw/Assert/Assert*
T0*#
_output_shapes
:?????????
~
&Predicates/BiLSTM/BiLSTM/bw/bw/Shape_2Shape(Predicates/BiLSTM/BiLSTM/bw/bw/transpose*
T0*
_output_shapes
:
~
4Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
6Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
6Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
.Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1StridedSlice&Predicates/BiLSTM/BiLSTM/bw/bw/Shape_24Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack6Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_16Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
~
&Predicates/BiLSTM/BiLSTM/bw/bw/Shape_3Shape(Predicates/BiLSTM/BiLSTM/bw/bw/transpose*
T0*
_output_shapes
:
~
4Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
6Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
?
6Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
.Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2StridedSlice&Predicates/BiLSTM/BiLSTM/bw/bw/Shape_34Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack6Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_16Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0
o
-Predicates/BiLSTM/BiLSTM/bw/bw/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
?
)Predicates/BiLSTM/BiLSTM/bw/bw/ExpandDims
ExpandDims.Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_2-Predicates/BiLSTM/BiLSTM/bw/bw/ExpandDims/dim*
_output_shapes
:*
T0
q
&Predicates/BiLSTM/BiLSTM/bw/bw/Const_1Const*
valueB:?*
_output_shapes
:*
dtype0
n
,Predicates/BiLSTM/BiLSTM/bw/bw/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
'Predicates/BiLSTM/BiLSTM/bw/bw/concat_1ConcatV2)Predicates/BiLSTM/BiLSTM/bw/bw/ExpandDims&Predicates/BiLSTM/BiLSTM/bw/bw/Const_1,Predicates/BiLSTM/BiLSTM/bw/bw/concat_1/axis*
N*
T0*
_output_shapes
:
o
*Predicates/BiLSTM/BiLSTM/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
$Predicates/BiLSTM/BiLSTM/bw/bw/zerosFill'Predicates/BiLSTM/BiLSTM/bw/bw/concat_1*Predicates/BiLSTM/BiLSTM/bw/bw/zeros/Const*
T0*(
_output_shapes
:??????????
p
&Predicates/BiLSTM/BiLSTM/bw/bw/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
?
"Predicates/BiLSTM/BiLSTM/bw/bw/MinMin*Predicates/BiLSTM/BiLSTM/bw/bw/CheckSeqLen&Predicates/BiLSTM/BiLSTM/bw/bw/Const_2*
T0*
_output_shapes
: 
p
&Predicates/BiLSTM/BiLSTM/bw/bw/Const_3Const*
dtype0*
valueB: *
_output_shapes
:
?
"Predicates/BiLSTM/BiLSTM/bw/bw/MaxMax*Predicates/BiLSTM/BiLSTM/bw/bw/CheckSeqLen&Predicates/BiLSTM/BiLSTM/bw/bw/Const_3*
T0*
_output_shapes
: 
e
#Predicates/BiLSTM/BiLSTM/bw/bw/timeConst*
dtype0*
value	B : *
_output_shapes
: 
?
*Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayTensorArrayV3.Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1*J
tensor_array_name53Predicates/BiLSTM/BiLSTM/bw/bw/dynamic_rnn/output_0*
identical_element_shapes(*
dtype0*%
element_shape:??????????*
_output_shapes

:: 
?
,Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray_1TensorArrayV3.Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1*
dtype0*$
element_shape:?????????<*I
tensor_array_name42Predicates/BiLSTM/BiLSTM/bw/bw/dynamic_rnn/input_0*
identical_element_shapes(*
_output_shapes

:: 
?
7Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/ShapeShape(Predicates/BiLSTM/BiLSTM/bw/bw/transpose*
T0*
_output_shapes
:
?
EPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
?
GPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
?
GPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
?Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice7Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/ShapeEPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stackGPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_1GPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 

=Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/startConst*
dtype0*
value	B : *
_output_shapes
: 

=Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
?
7Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/rangeRange=Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/start?Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/strided_slice=Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????
?
YPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3,Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray_17Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/range(Predicates/BiLSTM/BiLSTM/bw/bw/transpose.Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray_1:1*
T0*
_output_shapes
: *;
_class1
/-loc:@Predicates/BiLSTM/BiLSTM/bw/bw/transpose
j
(Predicates/BiLSTM/BiLSTM/bw/bw/Maximum/xConst*
dtype0*
value	B :*
_output_shapes
: 
?
&Predicates/BiLSTM/BiLSTM/bw/bw/MaximumMaximum(Predicates/BiLSTM/BiLSTM/bw/bw/Maximum/x"Predicates/BiLSTM/BiLSTM/bw/bw/Max*
T0*
_output_shapes
: 
?
&Predicates/BiLSTM/BiLSTM/bw/bw/MinimumMinimum.Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1&Predicates/BiLSTM/BiLSTM/bw/bw/Maximum*
_output_shapes
: *
T0
x
6Predicates/BiLSTM/BiLSTM/bw/bw/while/iteration_counterConst*
dtype0*
value	B : *
_output_shapes
: 
?
*Predicates/BiLSTM/BiLSTM/bw/bw/while/EnterEnter6Predicates/BiLSTM/BiLSTM/bw/bw/while/iteration_counter*
parallel_iterations *
T0*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: 
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_1Enter#Predicates/BiLSTM/BiLSTM/bw/bw/time*
_output_shapes
: *
T0*
parallel_iterations *B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_2Enter,Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray:1*
parallel_iterations *
T0*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: 
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_3EnterSPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros*
parallel_iterations *
T0*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*(
_output_shapes
:??????????
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_4EnterUPredicates/BiLSTM/BiLSTM/bw/bw/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1*
parallel_iterations *
T0*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*(
_output_shapes
:??????????
?
*Predicates/BiLSTM/BiLSTM/bw/bw/while/MergeMerge*Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter2Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration*
T0*
N*
_output_shapes
: : 
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_1Merge,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_14Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_1*
_output_shapes
: : *
T0*
N
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_2Merge,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_24Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_2*
_output_shapes
: : *
T0*
N
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_3Merge,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_34Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_3**
_output_shapes
:??????????: *
T0*
N
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_4Merge,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_44Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_4**
_output_shapes
:??????????: *
T0*
N
?
)Predicates/BiLSTM/BiLSTM/bw/bw/while/LessLess*Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge/Predicates/BiLSTM/BiLSTM/bw/bw/while/Less/Enter*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Less/EnterEnter.Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1*
is_constant(*
_output_shapes
: *
T0*
parallel_iterations *B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context
?
+Predicates/BiLSTM/BiLSTM/bw/bw/while/Less_1Less,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_11Predicates/BiLSTM/BiLSTM/bw/bw/while/Less_1/Enter*
_output_shapes
: *
T0
?
1Predicates/BiLSTM/BiLSTM/bw/bw/while/Less_1/EnterEnter&Predicates/BiLSTM/BiLSTM/bw/bw/Minimum*
is_constant(*
_output_shapes
: *
T0*
parallel_iterations *B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context
?
/Predicates/BiLSTM/BiLSTM/bw/bw/while/LogicalAnd
LogicalAnd)Predicates/BiLSTM/BiLSTM/bw/bw/while/Less+Predicates/BiLSTM/BiLSTM/bw/bw/while/Less_1*
_output_shapes
: 
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCondLoopCond/Predicates/BiLSTM/BiLSTM/bw/bw/while/LogicalAnd*
_output_shapes
: 
?
+Predicates/BiLSTM/BiLSTM/bw/bw/while/SwitchSwitch*Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge-Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_1Switch,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_1-Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_1
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_2Switch,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_2-Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_2
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_3Switch,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_3-Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*<
_output_shapes*
(:??????????:??????????*?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_3
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_4Switch,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_4-Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCond*
T0*<
_output_shapes*
(:??????????:??????????*?
_class5
31loc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_4
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/IdentityIdentity-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch:1*
T0*
_output_shapes
: 
?
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_1Identity/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_1:1*
_output_shapes
: *
T0
?
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_2Identity/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 
?
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_3Identity/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_3:1*
T0*(
_output_shapes
:??????????
?
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_4Identity/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_4:1*
T0*(
_output_shapes
:??????????
?
*Predicates/BiLSTM/BiLSTM/bw/bw/while/add/yConst.^Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
(Predicates/BiLSTM/BiLSTM/bw/bw/while/addAdd-Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity*Predicates/BiLSTM/BiLSTM/bw/bw/while/add/y*
_output_shapes
: *
T0
?
6Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3TensorArrayReadV3<Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_1>Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:?????????<*
dtype0
?
<Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/EnterEnter,Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray_1*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
parallel_iterations *
is_constant(*
_output_shapes
:*
T0
?
>Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1EnterYPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
parallel_iterations *
T0*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes
: 
?
1Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqualGreaterEqual/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_17Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Enter*#
_output_shapes
:?????????*
T0
?
7Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/EnterEnter*Predicates/BiLSTM/BiLSTM/bw/bw/CheckSeqLen*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
parallel_iterations *
is_constant(*#
_output_shapes
:?????????*
T0
?
FPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"?      *8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel*
_output_shapes
:*
dtype0
?
DPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *????*8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel*
_output_shapes
: *
dtype0
?
DPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *???=*
dtype0*
_output_shapes
: *8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel
?
NPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformFPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
T0* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel
?
DPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/subSubDPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/maxDPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel
?
DPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/mulMulNPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformDPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel
?
@Predicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniformAddDPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/mulDPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel
?
%Predicates/BiLSTM/bw/lstm_cell/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel
?
,Predicates/BiLSTM/bw/lstm_cell/kernel/AssignAssign%Predicates/BiLSTM/bw/lstm_cell/kernel@Predicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel
?
*Predicates/BiLSTM/bw/lstm_cell/kernel/readIdentity%Predicates/BiLSTM/bw/lstm_cell/kernel*
T0* 
_output_shapes
:
??
?
5Predicates/BiLSTM/bw/lstm_cell/bias/Initializer/ConstConst*
dtype0*
valueB?*    *
_output_shapes	
:?*6
_class,
*(loc:@Predicates/BiLSTM/bw/lstm_cell/bias
?
#Predicates/BiLSTM/bw/lstm_cell/bias
VariableV2*
shape:?*
_output_shapes	
:?*6
_class,
*(loc:@Predicates/BiLSTM/bw/lstm_cell/bias*
dtype0
?
*Predicates/BiLSTM/bw/lstm_cell/bias/AssignAssign#Predicates/BiLSTM/bw/lstm_cell/bias5Predicates/BiLSTM/bw/lstm_cell/bias/Initializer/Const*
T0*
_output_shapes	
:?*6
_class,
*(loc:@Predicates/BiLSTM/bw/lstm_cell/bias

(Predicates/BiLSTM/bw/lstm_cell/bias/readIdentity#Predicates/BiLSTM/bw/lstm_cell/bias*
T0*
_output_shapes	
:?
?
4Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zerosConst.^Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity*
_output_shapes	
:?*
valueB?*    *
dtype0
?
<Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCellLSTMBlockCell6Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_3/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_4BPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter4Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zeros4Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zeros4Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zerosDPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1*
T0*
	cell_clip%  ??*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????
?
BPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/EnterEnter*Predicates/BiLSTM/bw/lstm_cell/kernel/read*
parallel_iterations *
T0*
is_constant(*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context* 
_output_shapes
:
??
?
DPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1Enter(Predicates/BiLSTM/bw/lstm_cell/bias/read*
parallel_iterations *
T0*
is_constant(*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
_output_shapes	
:?
?
+Predicates/BiLSTM/BiLSTM/bw/bw/while/SelectSelect1Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual1Predicates/BiLSTM/BiLSTM/bw/bw/while/Select/Enter>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:6*(
_output_shapes
:??????????*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
?
1Predicates/BiLSTM/BiLSTM/bw/bw/while/Select/EnterEnter$Predicates/BiLSTM/BiLSTM/bw/bw/zeros*(
_output_shapes
:??????????*
parallel_iterations *O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
is_constant(*
T0
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Select_1Select1Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_3>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:1*(
_output_shapes
:??????????*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
?
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Select_2Select1Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_4>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:6*(
_output_shapes
:??????????*O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
?
HPredicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3NPredicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_1+Predicates/BiLSTM/BiLSTM/bw/bw/while/Select/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_2*
_output_shapes
: *O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
T0
?
NPredicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter*Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray*
_output_shapes
:*
parallel_iterations *O
_classE
CAloc:@Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell*
is_constant(*B

frame_name42Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context*
T0
?
,Predicates/BiLSTM/BiLSTM/bw/bw/while/add_1/yConst.^Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity*
_output_shapes
: *
dtype0*
value	B :
?
*Predicates/BiLSTM/BiLSTM/bw/bw/while/add_1Add/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_1,Predicates/BiLSTM/BiLSTM/bw/bw/while/add_1/y*
_output_shapes
: *
T0
?
2Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIterationNextIteration(Predicates/BiLSTM/BiLSTM/bw/bw/while/add*
_output_shapes
: *
T0
?
4Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_1NextIteration*Predicates/BiLSTM/BiLSTM/bw/bw/while/add_1*
_output_shapes
: *
T0
?
4Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_2NextIterationHPredicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
?
4Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_3NextIteration-Predicates/BiLSTM/BiLSTM/bw/bw/while/Select_1*(
_output_shapes
:??????????*
T0
?
4Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_4NextIteration-Predicates/BiLSTM/BiLSTM/bw/bw/while/Select_2*(
_output_shapes
:??????????*
T0

)Predicates/BiLSTM/BiLSTM/bw/bw/while/ExitExit+Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch*
_output_shapes
: *
T0
?
+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_1Exit-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_1*
_output_shapes
: *
T0
?
+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_2Exit-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_2*
_output_shapes
: *
T0
?
+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_3Exit-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_3*(
_output_shapes
:??????????*
T0
?
+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_4Exit-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_4*(
_output_shapes
:??????????*
T0
?
APredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3*Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_2*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray*
_output_shapes
: 
?
;Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/startConst*
_output_shapes
: *
value	B : *=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray*
dtype0
?
;Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/deltaConst*
_output_shapes
: *
value	B :*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray*
dtype0
?
5Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/rangeRange;Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/startAPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArraySizeV3;Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range/delta*#
_output_shapes
:?????????*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray
?
CPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3*Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray5Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/range+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_2*5
_output_shapes#
!:???????????????????*%
element_shape:??????????*=
_class3
1/loc:@Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray*
dtype0
q
&Predicates/BiLSTM/BiLSTM/bw/bw/Const_4Const*
_output_shapes
:*
valueB:?*
dtype0
g
%Predicates/BiLSTM/BiLSTM/bw/bw/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
n
,Predicates/BiLSTM/BiLSTM/bw/bw/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :
n
,Predicates/BiLSTM/BiLSTM/bw/bw/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
&Predicates/BiLSTM/BiLSTM/bw/bw/range_1Range,Predicates/BiLSTM/BiLSTM/bw/bw/range_1/start%Predicates/BiLSTM/BiLSTM/bw/bw/Rank_1,Predicates/BiLSTM/BiLSTM/bw/bw/range_1/delta*
_output_shapes
:
?
0Predicates/BiLSTM/BiLSTM/bw/bw/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
n
,Predicates/BiLSTM/BiLSTM/bw/bw/concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
'Predicates/BiLSTM/BiLSTM/bw/bw/concat_2ConcatV20Predicates/BiLSTM/BiLSTM/bw/bw/concat_2/values_0&Predicates/BiLSTM/BiLSTM/bw/bw/range_1,Predicates/BiLSTM/BiLSTM/bw/bw/concat_2/axis*
_output_shapes
:*
T0*
N
?
*Predicates/BiLSTM/BiLSTM/bw/bw/transpose_1	TransposeCPredicates/BiLSTM/BiLSTM/bw/bw/TensorArrayStack/TensorArrayGatherV3'Predicates/BiLSTM/BiLSTM/bw/bw/concat_2*
T0*5
_output_shapes#
!:???????????????????
?
!Predicates/BiLSTM/ReverseSequenceReverseSequence*Predicates/BiLSTM/BiLSTM/bw/bw/transpose_1pred_original_sequence_lengths*
seq_dim*

Tlen0*5
_output_shapes#
!:???????????????????*
T0
[
Predicates/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/ExpandDims
ExpandDims*Predicates/BiLSTM/BiLSTM/fw/fw/transpose_1Predicates/ExpandDims/dim*9
_output_shapes'
%:#???????????????????*
T0
]
Predicates/ExpandDims_1/dimConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/ExpandDims_1
ExpandDims!Predicates/BiLSTM/ReverseSequencePredicates/ExpandDims_1/dim*9
_output_shapes'
%:#???????????????????*
T0
c
Predicates/concat_2/axisConst*
valueB :
?????????*
_output_shapes
: *
dtype0
?
Predicates/concat_2ConcatV2Predicates/ExpandDimsPredicates/ExpandDims_1Predicates/concat_2/axis*9
_output_shapes'
%:#???????????????????*
T0*
N
l
!Predicates/Mean/reduction_indicesConst*
dtype0*
valueB :
?????????*
_output_shapes
: 
?
Predicates/MeanMeanPredicates/concat_2!Predicates/Mean/reduction_indices*
T0*5
_output_shapes#
!:???????????????????
?
1Predicates/proj/W/Initializer/random_normal/shapeConst*$
_class
loc:@Predicates/proj/W*
valueB"?      *
dtype0*
_output_shapes
:
?
0Predicates/proj/W/Initializer/random_normal/meanConst*$
_class
loc:@Predicates/proj/W*
valueB
 *    *
dtype0*
_output_shapes
: 
?
2Predicates/proj/W/Initializer/random_normal/stddevConst*$
_class
loc:@Predicates/proj/W*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
@Predicates/proj/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal1Predicates/proj/W/Initializer/random_normal/shape*$
_class
loc:@Predicates/proj/W*
dtype0*
T0*
_output_shapes
:	?
?
/Predicates/proj/W/Initializer/random_normal/mulMul@Predicates/proj/W/Initializer/random_normal/RandomStandardNormal2Predicates/proj/W/Initializer/random_normal/stddev*$
_class
loc:@Predicates/proj/W*
T0*
_output_shapes
:	?
?
+Predicates/proj/W/Initializer/random_normalAdd/Predicates/proj/W/Initializer/random_normal/mul0Predicates/proj/W/Initializer/random_normal/mean*
_output_shapes
:	?*
T0*$
_class
loc:@Predicates/proj/W
?
 Predicates/proj/W/Initializer/QrQr+Predicates/proj/W/Initializer/random_normal*$
_class
loc:@Predicates/proj/W*
T0*)
_output_shapes
:	?:
?
&Predicates/proj/W/Initializer/DiagPartDiagPart"Predicates/proj/W/Initializer/Qr:1*$
_class
loc:@Predicates/proj/W*
T0*
_output_shapes
:
?
"Predicates/proj/W/Initializer/SignSign&Predicates/proj/W/Initializer/DiagPart*
_output_shapes
:*
T0*$
_class
loc:@Predicates/proj/W
?
!Predicates/proj/W/Initializer/mulMul Predicates/proj/W/Initializer/Qr"Predicates/proj/W/Initializer/Sign*
_output_shapes
:	?*
T0*$
_class
loc:@Predicates/proj/W
?
+Predicates/proj/W/Initializer/Reshape/shapeConst*
_output_shapes
:*
dtype0*$
_class
loc:@Predicates/proj/W*
valueB"?      
?
%Predicates/proj/W/Initializer/ReshapeReshape!Predicates/proj/W/Initializer/mul+Predicates/proj/W/Initializer/Reshape/shape*
_output_shapes
:	?*
T0*$
_class
loc:@Predicates/proj/W
?
%Predicates/proj/W/Initializer/mul_1/xConst*
_output_shapes
: *
dtype0*$
_class
loc:@Predicates/proj/W*
valueB
 *???
?
#Predicates/proj/W/Initializer/mul_1Mul%Predicates/proj/W/Initializer/mul_1/x%Predicates/proj/W/Initializer/Reshape*
_output_shapes
:	?*
T0*$
_class
loc:@Predicates/proj/W
?
Predicates/proj/W
VariableV2*
shape:	?*
_output_shapes
:	?*
dtype0*$
_class
loc:@Predicates/proj/W
?
Predicates/proj/W/AssignAssignPredicates/proj/W#Predicates/proj/W/Initializer/mul_1*
_output_shapes
:	?*
T0*$
_class
loc:@Predicates/proj/W
?
Predicates/proj/W/readIdentityPredicates/proj/W*
_output_shapes
:	?*$
_class
loc:@Predicates/proj/W*
T0
?
#Predicates/proj/b/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@Predicates/proj/b*
valueB*    *
dtype0

Predicates/proj/b
VariableV2*$
_class
loc:@Predicates/proj/b*
_output_shapes
:*
shape:*
dtype0
?
Predicates/proj/b/AssignAssignPredicates/proj/b#Predicates/proj/b/Initializer/zeros*
_output_shapes
:*$
_class
loc:@Predicates/proj/b*
T0
?
Predicates/proj/b/readIdentityPredicates/proj/b*
_output_shapes
:*$
_class
loc:@Predicates/proj/b*
T0
n
Predicates/proj/Reshape/shapeConst*
_output_shapes
:*
valueB"?????   *
dtype0
?
Predicates/proj/ReshapeReshapePredicates/MeanPredicates/proj/Reshape/shape*(
_output_shapes
:??????????*
T0
?
Predicates/proj/MatMulMatMulPredicates/proj/ReshapePredicates/proj/W/read*'
_output_shapes
:?????????*
T0
|
Predicates/proj/addAddPredicates/proj/MatMulPredicates/proj/b/read*'
_output_shapes
:?????????*
T0
t
Predicates/proj/Reshape_1/shapeConst*
_output_shapes
:*!
valueB"?????      *
dtype0
?
Predicates/proj/Reshape_1ReshapePredicates/proj/addPredicates/proj/Reshape_1/shape*,
_output_shapes
:??????????*
T0
?
:Predicates/CRF/transition/Initializer/random_uniform/shapeConst*
_output_shapes
:*,
_class"
 loc:@Predicates/CRF/transition*
dtype0*
valueB"      
?
8Predicates/CRF/transition/Initializer/random_uniform/minConst*
_output_shapes
: *,
_class"
 loc:@Predicates/CRF/transition*
dtype0*
valueB
 *b?'?
?
8Predicates/CRF/transition/Initializer/random_uniform/maxConst*
_output_shapes
: *,
_class"
 loc:@Predicates/CRF/transition*
dtype0*
valueB
 *b?'?
?
BPredicates/CRF/transition/Initializer/random_uniform/RandomUniformRandomUniform:Predicates/CRF/transition/Initializer/random_uniform/shape*
_output_shapes

:*,
_class"
 loc:@Predicates/CRF/transition*
dtype0*
T0
?
8Predicates/CRF/transition/Initializer/random_uniform/subSub8Predicates/CRF/transition/Initializer/random_uniform/max8Predicates/CRF/transition/Initializer/random_uniform/min*
_output_shapes
: *,
_class"
 loc:@Predicates/CRF/transition*
T0
?
8Predicates/CRF/transition/Initializer/random_uniform/mulMulBPredicates/CRF/transition/Initializer/random_uniform/RandomUniform8Predicates/CRF/transition/Initializer/random_uniform/sub*
_output_shapes

:*,
_class"
 loc:@Predicates/CRF/transition*
T0
?
4Predicates/CRF/transition/Initializer/random_uniformAdd8Predicates/CRF/transition/Initializer/random_uniform/mul8Predicates/CRF/transition/Initializer/random_uniform/min*
_output_shapes

:*,
_class"
 loc:@Predicates/CRF/transition*
T0
?
Predicates/CRF/transition
VariableV2*
_output_shapes

:*,
_class"
 loc:@Predicates/CRF/transition*
dtype0*
shape
:
?
 Predicates/CRF/transition/AssignAssignPredicates/CRF/transition4Predicates/CRF/transition/Initializer/random_uniform*
_output_shapes

:*,
_class"
 loc:@Predicates/CRF/transition*
T0
?
Predicates/CRF/transition/readIdentityPredicates/CRF/transition*
T0*,
_class"
 loc:@Predicates/CRF/transition*
_output_shapes

:
Y
Predicates/CRF/Equal/xConst*
dtype0*
value
B :?*
_output_shapes
: 
X
Predicates/CRF/Equal/yConst*
dtype0*
value	B :*
_output_shapes
: 
n
Predicates/CRF/EqualEqualPredicates/CRF/Equal/xPredicates/CRF/Equal/y*
T0*
_output_shapes
: 
_
Predicates/CRF/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
Predicates/CRF/ExpandDims
ExpandDimsPredicates/CRF/transition/readPredicates/CRF/ExpandDims/dim*
T0*"
_output_shapes
:
o
Predicates/CRF/Slice/beginConst*
dtype0*!
valueB"            *
_output_shapes
:
n
Predicates/CRF/Slice/sizeConst*
dtype0*!
valueB"????   ????*
_output_shapes
:
?
Predicates/CRF/SliceSlicePredicates/proj/Reshape_1Predicates/CRF/Slice/beginPredicates/CRF/Slice/size*
T0*
Index0*+
_output_shapes
:?????????
?
Predicates/CRF/SqueezeSqueezePredicates/CRF/Slice*
T0*
squeeze_dims
*'
_output_shapes
:?????????
q
Predicates/CRF/Slice_1/beginConst*
dtype0*!
valueB"           *
_output_shapes
:
p
Predicates/CRF/Slice_1/sizeConst*
dtype0*!
valueB"????????????*
_output_shapes
:
?
Predicates/CRF/Slice_1SlicePredicates/proj/Reshape_1Predicates/CRF/Slice_1/beginPredicates/CRF/Slice_1/size*
T0*
Index0*,
_output_shapes
:??????????
V
Predicates/CRF/ConstConst*
dtype0*
value	B : *
_output_shapes
: 
V
Predicates/CRF/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
}
Predicates/CRF/subSubpred_original_sequence_lengthsPredicates/CRF/sub/y*
T0*#
_output_shapes
:?????????
y
Predicates/CRF/MaximumMaximumPredicates/CRF/ConstPredicates/CRF/sub*
T0*#
_output_shapes
:?????????
Y
Predicates/CRF/rnn/RankConst*
dtype0*
value	B :*
_output_shapes
: 
`
Predicates/CRF/rnn/range/startConst*
dtype0*
value	B :*
_output_shapes
: 
`
Predicates/CRF/rnn/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
?
Predicates/CRF/rnn/rangeRangePredicates/CRF/rnn/range/startPredicates/CRF/rnn/RankPredicates/CRF/rnn/range/delta*
_output_shapes
:
s
"Predicates/CRF/rnn/concat/values_0Const*
dtype0*
valueB"       *
_output_shapes
:
`
Predicates/CRF/rnn/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
Predicates/CRF/rnn/concatConcatV2"Predicates/CRF/rnn/concat/values_0Predicates/CRF/rnn/rangePredicates/CRF/rnn/concat/axis*
N*
T0*
_output_shapes
:
?
Predicates/CRF/rnn/transpose	TransposePredicates/CRF/Slice_1Predicates/CRF/rnn/concat*
T0*,
_output_shapes
:??????????
t
"Predicates/CRF/rnn/sequence_lengthIdentityPredicates/CRF/Maximum*
T0*#
_output_shapes
:?????????
d
Predicates/CRF/rnn/ShapeShapePredicates/CRF/rnn/transpose*
T0*
_output_shapes
:
p
&Predicates/CRF/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
r
(Predicates/CRF/rnn/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
r
(Predicates/CRF/rnn/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
 Predicates/CRF/rnn/strided_sliceStridedSlicePredicates/CRF/rnn/Shape&Predicates/CRF/rnn/strided_slice/stack(Predicates/CRF/rnn/strided_slice/stack_1(Predicates/CRF/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
l
Predicates/CRF/rnn/Shape_1Shape"Predicates/CRF/rnn/sequence_length*
T0*
_output_shapes
:
p
Predicates/CRF/rnn/stackPack Predicates/CRF/rnn/strided_slice*
N*
T0*
_output_shapes
:
|
Predicates/CRF/rnn/EqualEqualPredicates/CRF/rnn/Shape_1Predicates/CRF/rnn/stack*
T0*
_output_shapes
:
b
Predicates/CRF/rnn/ConstConst*
dtype0*
valueB: *
_output_shapes
:
i
Predicates/CRF/rnn/AllAllPredicates/CRF/rnn/EqualPredicates/CRF/rnn/Const*
_output_shapes
: 
?
Predicates/CRF/rnn/Assert/ConstConst*S
valueJBH BBExpected shape for Tensor Predicates/CRF/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
r
!Predicates/CRF/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
?
'Predicates/CRF/rnn/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBExpected shape for Tensor Predicates/CRF/rnn/sequence_length:0 is 
x
'Predicates/CRF/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
?
 Predicates/CRF/rnn/Assert/AssertAssertPredicates/CRF/rnn/All'Predicates/CRF/rnn/Assert/Assert/data_0Predicates/CRF/rnn/stack'Predicates/CRF/rnn/Assert/Assert/data_2Predicates/CRF/rnn/Shape_1*
T
2
?
Predicates/CRF/rnn/CheckSeqLenIdentity"Predicates/CRF/rnn/sequence_length!^Predicates/CRF/rnn/Assert/Assert*
T0*#
_output_shapes
:?????????
f
Predicates/CRF/rnn/Shape_2ShapePredicates/CRF/rnn/transpose*
_output_shapes
:*
T0
r
(Predicates/CRF/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
t
*Predicates/CRF/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
t
*Predicates/CRF/rnn/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
"Predicates/CRF/rnn/strided_slice_1StridedSlicePredicates/CRF/rnn/Shape_2(Predicates/CRF/rnn/strided_slice_1/stack*Predicates/CRF/rnn/strided_slice_1/stack_1*Predicates/CRF/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
f
Predicates/CRF/rnn/Shape_3ShapePredicates/CRF/rnn/transpose*
_output_shapes
:*
T0
r
(Predicates/CRF/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
t
*Predicates/CRF/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
t
*Predicates/CRF/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
"Predicates/CRF/rnn/strided_slice_2StridedSlicePredicates/CRF/rnn/Shape_3(Predicates/CRF/rnn/strided_slice_2/stack*Predicates/CRF/rnn/strided_slice_2/stack_1*Predicates/CRF/rnn/strided_slice_2/stack_2*
_output_shapes
: *
T0*
shrink_axis_mask*
Index0
c
!Predicates/CRF/rnn/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
Predicates/CRF/rnn/ExpandDims
ExpandDims"Predicates/CRF/rnn/strided_slice_2!Predicates/CRF/rnn/ExpandDims/dim*
_output_shapes
:*
T0
d
Predicates/CRF/rnn/Const_1Const*
_output_shapes
:*
dtype0*
valueB:
b
 Predicates/CRF/rnn/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Predicates/CRF/rnn/concat_1ConcatV2Predicates/CRF/rnn/ExpandDimsPredicates/CRF/rnn/Const_1 Predicates/CRF/rnn/concat_1/axis*
_output_shapes
:*
T0*
N
`
Predicates/CRF/rnn/zeros/ConstConst*
_output_shapes
: *
value	B : *
dtype0
?
Predicates/CRF/rnn/zerosFillPredicates/CRF/rnn/concat_1Predicates/CRF/rnn/zeros/Const*
T0*'
_output_shapes
:?????????
d
Predicates/CRF/rnn/Const_2Const*
_output_shapes
:*
valueB: *
dtype0
z
Predicates/CRF/rnn/MinMinPredicates/CRF/rnn/CheckSeqLenPredicates/CRF/rnn/Const_2*
T0*
_output_shapes
: 
d
Predicates/CRF/rnn/Const_3Const*
_output_shapes
:*
valueB: *
dtype0
z
Predicates/CRF/rnn/MaxMaxPredicates/CRF/rnn/CheckSeqLenPredicates/CRF/rnn/Const_3*
T0*
_output_shapes
: 
Y
Predicates/CRF/rnn/timeConst*
_output_shapes
: *
value	B : *
dtype0
?
Predicates/CRF/rnn/TensorArrayTensorArrayV3"Predicates/CRF/rnn/strided_slice_1*
_output_shapes

:: *>
tensor_array_name)'Predicates/CRF/rnn/dynamic_rnn/output_0*$
element_shape:?????????*
dtype0*
identical_element_shapes(
?
 Predicates/CRF/rnn/TensorArray_1TensorArrayV3"Predicates/CRF/rnn/strided_slice_1*
_output_shapes

:: *=
tensor_array_name(&Predicates/CRF/rnn/dynamic_rnn/input_0*$
element_shape:?????????*
dtype0*
identical_element_shapes(
w
+Predicates/CRF/rnn/TensorArrayUnstack/ShapeShapePredicates/CRF/rnn/transpose*
T0*
_output_shapes
:
?
9Predicates/CRF/rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
;Predicates/CRF/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
?
;Predicates/CRF/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
3Predicates/CRF/rnn/TensorArrayUnstack/strided_sliceStridedSlice+Predicates/CRF/rnn/TensorArrayUnstack/Shape9Predicates/CRF/rnn/TensorArrayUnstack/strided_slice/stack;Predicates/CRF/rnn/TensorArrayUnstack/strided_slice/stack_1;Predicates/CRF/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0
s
1Predicates/CRF/rnn/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
s
1Predicates/CRF/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
+Predicates/CRF/rnn/TensorArrayUnstack/rangeRange1Predicates/CRF/rnn/TensorArrayUnstack/range/start3Predicates/CRF/rnn/TensorArrayUnstack/strided_slice1Predicates/CRF/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????
?
MPredicates/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3 Predicates/CRF/rnn/TensorArray_1+Predicates/CRF/rnn/TensorArrayUnstack/rangePredicates/CRF/rnn/transpose"Predicates/CRF/rnn/TensorArray_1:1*
T0*
_output_shapes
: */
_class%
#!loc:@Predicates/CRF/rnn/transpose
^
Predicates/CRF/rnn/Maximum/xConst*
value	B :*
_output_shapes
: *
dtype0
|
Predicates/CRF/rnn/MaximumMaximumPredicates/CRF/rnn/Maximum/xPredicates/CRF/rnn/Max*
_output_shapes
: *
T0
?
Predicates/CRF/rnn/MinimumMinimum"Predicates/CRF/rnn/strided_slice_1Predicates/CRF/rnn/Maximum*
_output_shapes
: *
T0
l
*Predicates/CRF/rnn/while/iteration_counterConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn/while/EnterEnter*Predicates/CRF/rnn/while/iteration_counter*6

frame_name(&Predicates/CRF/rnn/while/while_context*
parallel_iterations *
_output_shapes
: *
T0
?
 Predicates/CRF/rnn/while/Enter_1EnterPredicates/CRF/rnn/time*6

frame_name(&Predicates/CRF/rnn/while/while_context*
parallel_iterations *
_output_shapes
: *
T0
?
 Predicates/CRF/rnn/while/Enter_2Enter Predicates/CRF/rnn/TensorArray:1*
parallel_iterations *
T0*6

frame_name(&Predicates/CRF/rnn/while/while_context*
_output_shapes
: 
?
 Predicates/CRF/rnn/while/Enter_3EnterPredicates/CRF/Squeeze*
parallel_iterations *
T0*6

frame_name(&Predicates/CRF/rnn/while/while_context*'
_output_shapes
:?????????
?
Predicates/CRF/rnn/while/MergeMergePredicates/CRF/rnn/while/Enter&Predicates/CRF/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
?
 Predicates/CRF/rnn/while/Merge_1Merge Predicates/CRF/rnn/while/Enter_1(Predicates/CRF/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
?
 Predicates/CRF/rnn/while/Merge_2Merge Predicates/CRF/rnn/while/Enter_2(Predicates/CRF/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
?
 Predicates/CRF/rnn/while/Merge_3Merge Predicates/CRF/rnn/while/Enter_3(Predicates/CRF/rnn/while/NextIteration_3*
T0*
N*)
_output_shapes
:?????????: 
?
Predicates/CRF/rnn/while/LessLessPredicates/CRF/rnn/while/Merge#Predicates/CRF/rnn/while/Less/Enter*
T0*
_output_shapes
: 
?
#Predicates/CRF/rnn/while/Less/EnterEnter"Predicates/CRF/rnn/strided_slice_1*
is_constant(*
parallel_iterations *
T0*6

frame_name(&Predicates/CRF/rnn/while/while_context*
_output_shapes
: 
?
Predicates/CRF/rnn/while/Less_1Less Predicates/CRF/rnn/while/Merge_1%Predicates/CRF/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
?
%Predicates/CRF/rnn/while/Less_1/EnterEnterPredicates/CRF/rnn/Minimum*
parallel_iterations *
T0*
is_constant(*6

frame_name(&Predicates/CRF/rnn/while/while_context*
_output_shapes
: 
?
#Predicates/CRF/rnn/while/LogicalAnd
LogicalAndPredicates/CRF/rnn/while/LessPredicates/CRF/rnn/while/Less_1*
_output_shapes
: 
j
!Predicates/CRF/rnn/while/LoopCondLoopCond#Predicates/CRF/rnn/while/LogicalAnd*
_output_shapes
: 
?
Predicates/CRF/rnn/while/SwitchSwitchPredicates/CRF/rnn/while/Merge!Predicates/CRF/rnn/while/LoopCond*
T0*
_output_shapes
: : *1
_class'
%#loc:@Predicates/CRF/rnn/while/Merge
?
!Predicates/CRF/rnn/while/Switch_1Switch Predicates/CRF/rnn/while/Merge_1!Predicates/CRF/rnn/while/LoopCond*
T0*3
_class)
'%loc:@Predicates/CRF/rnn/while/Merge_1*
_output_shapes
: : 
?
!Predicates/CRF/rnn/while/Switch_2Switch Predicates/CRF/rnn/while/Merge_2!Predicates/CRF/rnn/while/LoopCond*
T0*3
_class)
'%loc:@Predicates/CRF/rnn/while/Merge_2*
_output_shapes
: : 
?
!Predicates/CRF/rnn/while/Switch_3Switch Predicates/CRF/rnn/while/Merge_3!Predicates/CRF/rnn/while/LoopCond*
T0*3
_class)
'%loc:@Predicates/CRF/rnn/while/Merge_3*:
_output_shapes(
&:?????????:?????????
q
!Predicates/CRF/rnn/while/IdentityIdentity!Predicates/CRF/rnn/while/Switch:1*
T0*
_output_shapes
: 
u
#Predicates/CRF/rnn/while/Identity_1Identity#Predicates/CRF/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
u
#Predicates/CRF/rnn/while/Identity_2Identity#Predicates/CRF/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
?
#Predicates/CRF/rnn/while/Identity_3Identity#Predicates/CRF/rnn/while/Switch_3:1*
T0*'
_output_shapes
:?????????
?
Predicates/CRF/rnn/while/add/yConst"^Predicates/CRF/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn/while/addAdd!Predicates/CRF/rnn/while/IdentityPredicates/CRF/rnn/while/add/y*
T0*
_output_shapes
: 
?
*Predicates/CRF/rnn/while/TensorArrayReadV3TensorArrayReadV30Predicates/CRF/rnn/while/TensorArrayReadV3/Enter#Predicates/CRF/rnn/while/Identity_12Predicates/CRF/rnn/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:?????????*
dtype0
?
0Predicates/CRF/rnn/while/TensorArrayReadV3/EnterEnter Predicates/CRF/rnn/TensorArray_1*
is_constant(*
parallel_iterations *
T0*
_output_shapes
:*6

frame_name(&Predicates/CRF/rnn/while/while_context
?
2Predicates/CRF/rnn/while/TensorArrayReadV3/Enter_1EnterMPredicates/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
parallel_iterations *
T0*
_output_shapes
: *6

frame_name(&Predicates/CRF/rnn/while/while_context
?
%Predicates/CRF/rnn/while/GreaterEqualGreaterEqual#Predicates/CRF/rnn/while/Identity_1+Predicates/CRF/rnn/while/GreaterEqual/Enter*
T0*#
_output_shapes
:?????????
?
+Predicates/CRF/rnn/while/GreaterEqual/EnterEnterPredicates/CRF/rnn/CheckSeqLen*
is_constant(*
parallel_iterations *
T0*#
_output_shapes
:?????????*6

frame_name(&Predicates/CRF/rnn/while/while_context
?
'Predicates/CRF/rnn/while/ExpandDims/dimConst"^Predicates/CRF/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
#Predicates/CRF/rnn/while/ExpandDims
ExpandDims#Predicates/CRF/rnn/while/Identity_3'Predicates/CRF/rnn/while/ExpandDims/dim*+
_output_shapes
:?????????*
T0
?
Predicates/CRF/rnn/while/add_1Add#Predicates/CRF/rnn/while/ExpandDims$Predicates/CRF/rnn/while/add_1/Enter*+
_output_shapes
:?????????*
T0
?
$Predicates/CRF/rnn/while/add_1/EnterEnterPredicates/CRF/ExpandDims*
is_constant(*"
_output_shapes
:*
T0*6

frame_name(&Predicates/CRF/rnn/while/while_context*
parallel_iterations 
?
.Predicates/CRF/rnn/while/Max/reduction_indicesConst"^Predicates/CRF/rnn/while/Identity*
valueB:*
_output_shapes
:*
dtype0
?
Predicates/CRF/rnn/while/MaxMaxPredicates/CRF/rnn/while/add_1.Predicates/CRF/rnn/while/Max/reduction_indices*'
_output_shapes
:?????????*
T0
?
Predicates/CRF/rnn/while/add_2Add*Predicates/CRF/rnn/while/TensorArrayReadV3Predicates/CRF/rnn/while/Max*'
_output_shapes
:?????????*
T0
?
)Predicates/CRF/rnn/while/ArgMax/dimensionConst"^Predicates/CRF/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn/while/ArgMaxArgMaxPredicates/CRF/rnn/while/add_1)Predicates/CRF/rnn/while/ArgMax/dimension*'
_output_shapes
:?????????*
T0
?
Predicates/CRF/rnn/while/CastCastPredicates/CRF/rnn/while/ArgMax*'
_output_shapes
:?????????*

SrcT0	*

DstT0
?
Predicates/CRF/rnn/while/SelectSelect%Predicates/CRF/rnn/while/GreaterEqual%Predicates/CRF/rnn/while/Select/EnterPredicates/CRF/rnn/while/Cast*
T0*'
_output_shapes
:?????????*0
_class&
$"loc:@Predicates/CRF/rnn/while/Cast
?
%Predicates/CRF/rnn/while/Select/EnterEnterPredicates/CRF/rnn/zeros*'
_output_shapes
:?????????*6

frame_name(&Predicates/CRF/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations *0
_class&
$"loc:@Predicates/CRF/rnn/while/Cast
?
!Predicates/CRF/rnn/while/Select_1Select%Predicates/CRF/rnn/while/GreaterEqual#Predicates/CRF/rnn/while/Identity_3Predicates/CRF/rnn/while/add_2*
T0*'
_output_shapes
:?????????*1
_class'
%#loc:@Predicates/CRF/rnn/while/add_2
?
<Predicates/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3BPredicates/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter#Predicates/CRF/rnn/while/Identity_1Predicates/CRF/rnn/while/Select#Predicates/CRF/rnn/while/Identity_2*
T0*
_output_shapes
: *0
_class&
$"loc:@Predicates/CRF/rnn/while/Cast
?
BPredicates/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterPredicates/CRF/rnn/TensorArray*
is_constant(*6

frame_name(&Predicates/CRF/rnn/while/while_context*
T0*0
_class&
$"loc:@Predicates/CRF/rnn/while/Cast*
parallel_iterations *
_output_shapes
:
?
 Predicates/CRF/rnn/while/add_3/yConst"^Predicates/CRF/rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn/while/add_3Add#Predicates/CRF/rnn/while/Identity_1 Predicates/CRF/rnn/while/add_3/y*
T0*
_output_shapes
: 
v
&Predicates/CRF/rnn/while/NextIterationNextIterationPredicates/CRF/rnn/while/add*
_output_shapes
: *
T0
z
(Predicates/CRF/rnn/while/NextIteration_1NextIterationPredicates/CRF/rnn/while/add_3*
_output_shapes
: *
T0
?
(Predicates/CRF/rnn/while/NextIteration_2NextIteration<Predicates/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
?
(Predicates/CRF/rnn/while/NextIteration_3NextIteration!Predicates/CRF/rnn/while/Select_1*'
_output_shapes
:?????????*
T0
g
Predicates/CRF/rnn/while/ExitExitPredicates/CRF/rnn/while/Switch*
_output_shapes
: *
T0
k
Predicates/CRF/rnn/while/Exit_1Exit!Predicates/CRF/rnn/while/Switch_1*
_output_shapes
: *
T0
k
Predicates/CRF/rnn/while/Exit_2Exit!Predicates/CRF/rnn/while/Switch_2*
_output_shapes
: *
T0
|
Predicates/CRF/rnn/while/Exit_3Exit!Predicates/CRF/rnn/while/Switch_3*'
_output_shapes
:?????????*
T0
?
5Predicates/CRF/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3Predicates/CRF/rnn/TensorArrayPredicates/CRF/rnn/while/Exit_2*
_output_shapes
: *1
_class'
%#loc:@Predicates/CRF/rnn/TensorArray
?
/Predicates/CRF/rnn/TensorArrayStack/range/startConst*
_output_shapes
: *1
_class'
%#loc:@Predicates/CRF/rnn/TensorArray*
dtype0*
value	B : 
?
/Predicates/CRF/rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: *1
_class'
%#loc:@Predicates/CRF/rnn/TensorArray*
dtype0*
value	B :
?
)Predicates/CRF/rnn/TensorArrayStack/rangeRange/Predicates/CRF/rnn/TensorArrayStack/range/start5Predicates/CRF/rnn/TensorArrayStack/TensorArraySizeV3/Predicates/CRF/rnn/TensorArrayStack/range/delta*#
_output_shapes
:?????????*1
_class'
%#loc:@Predicates/CRF/rnn/TensorArray
?
7Predicates/CRF/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3Predicates/CRF/rnn/TensorArray)Predicates/CRF/rnn/TensorArrayStack/rangePredicates/CRF/rnn/while/Exit_2*$
element_shape:?????????*1
_class'
%#loc:@Predicates/CRF/rnn/TensorArray*
dtype0*,
_output_shapes
:??????????
d
Predicates/CRF/rnn/Const_4Const*
dtype0*
valueB:*
_output_shapes
:
[
Predicates/CRF/rnn/Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
b
 Predicates/CRF/rnn/range_1/startConst*
dtype0*
value	B :*
_output_shapes
: 
b
 Predicates/CRF/rnn/range_1/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
?
Predicates/CRF/rnn/range_1Range Predicates/CRF/rnn/range_1/startPredicates/CRF/rnn/Rank_1 Predicates/CRF/rnn/range_1/delta*
_output_shapes
:
u
$Predicates/CRF/rnn/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
b
 Predicates/CRF/rnn/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
Predicates/CRF/rnn/concat_2ConcatV2$Predicates/CRF/rnn/concat_2/values_0Predicates/CRF/rnn/range_1 Predicates/CRF/rnn/concat_2/axis*
T0*
N*
_output_shapes
:
?
Predicates/CRF/rnn/transpose_1	Transpose7Predicates/CRF/rnn/TensorArrayStack/TensorArrayGatherV3Predicates/CRF/rnn/concat_2*
T0*,
_output_shapes
:??????????
?
Predicates/CRF/ReverseSequenceReverseSequencePredicates/CRF/rnn/transpose_1Predicates/CRF/Maximum*,
_output_shapes
:??????????*

Tlen0*
T0*
seq_dim
a
Predicates/CRF/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
?
Predicates/CRF/ArgMaxArgMaxPredicates/CRF/rnn/while/Exit_3Predicates/CRF/ArgMax/dimension*
T0*#
_output_shapes
:?????????
o
Predicates/CRF/CastCastPredicates/CRF/ArgMax*

DstT0*#
_output_shapes
:?????????*

SrcT0	
j
Predicates/CRF/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Predicates/CRF/ExpandDims_1
ExpandDimsPredicates/CRF/CastPredicates/CRF/ExpandDims_1/dim*
T0*'
_output_shapes
:?????????
[
Predicates/CRF/rnn_1/RankConst*
_output_shapes
: *
dtype0*
value	B :
b
 Predicates/CRF/rnn_1/range/startConst*
_output_shapes
: *
dtype0*
value	B :
b
 Predicates/CRF/rnn_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Predicates/CRF/rnn_1/rangeRange Predicates/CRF/rnn_1/range/startPredicates/CRF/rnn_1/Rank Predicates/CRF/rnn_1/range/delta*
_output_shapes
:
u
$Predicates/CRF/rnn_1/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
b
 Predicates/CRF/rnn_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Predicates/CRF/rnn_1/concatConcatV2$Predicates/CRF/rnn_1/concat/values_0Predicates/CRF/rnn_1/range Predicates/CRF/rnn_1/concat/axis*
N*
_output_shapes
:*
T0
?
Predicates/CRF/rnn_1/transpose	TransposePredicates/CRF/ReverseSequencePredicates/CRF/rnn_1/concat*,
_output_shapes
:??????????*
T0
v
$Predicates/CRF/rnn_1/sequence_lengthIdentityPredicates/CRF/Maximum*#
_output_shapes
:?????????*
T0
h
Predicates/CRF/rnn_1/ShapeShapePredicates/CRF/rnn_1/transpose*
_output_shapes
:*
T0
r
(Predicates/CRF/rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
t
*Predicates/CRF/rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
t
*Predicates/CRF/rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
"Predicates/CRF/rnn_1/strided_sliceStridedSlicePredicates/CRF/rnn_1/Shape(Predicates/CRF/rnn_1/strided_slice/stack*Predicates/CRF/rnn_1/strided_slice/stack_1*Predicates/CRF/rnn_1/strided_slice/stack_2*
_output_shapes
: *
T0*
shrink_axis_mask*
Index0
p
Predicates/CRF/rnn_1/Shape_1Shape$Predicates/CRF/rnn_1/sequence_length*
_output_shapes
:*
T0
t
Predicates/CRF/rnn_1/stackPack"Predicates/CRF/rnn_1/strided_slice*
N*
T0*
_output_shapes
:
?
Predicates/CRF/rnn_1/EqualEqualPredicates/CRF/rnn_1/Shape_1Predicates/CRF/rnn_1/stack*
_output_shapes
:*
T0
d
Predicates/CRF/rnn_1/ConstConst*
dtype0*
valueB: *
_output_shapes
:
o
Predicates/CRF/rnn_1/AllAllPredicates/CRF/rnn_1/EqualPredicates/CRF/rnn_1/Const*
_output_shapes
: 
?
!Predicates/CRF/rnn_1/Assert/ConstConst*
dtype0*U
valueLBJ BDExpected shape for Tensor Predicates/CRF/rnn_1/sequence_length:0 is *
_output_shapes
: 
t
#Predicates/CRF/rnn_1/Assert/Const_1Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
?
)Predicates/CRF/rnn_1/Assert/Assert/data_0Const*
dtype0*U
valueLBJ BDExpected shape for Tensor Predicates/CRF/rnn_1/sequence_length:0 is *
_output_shapes
: 
z
)Predicates/CRF/rnn_1/Assert/Assert/data_2Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
?
"Predicates/CRF/rnn_1/Assert/AssertAssertPredicates/CRF/rnn_1/All)Predicates/CRF/rnn_1/Assert/Assert/data_0Predicates/CRF/rnn_1/stack)Predicates/CRF/rnn_1/Assert/Assert/data_2Predicates/CRF/rnn_1/Shape_1*
T
2
?
 Predicates/CRF/rnn_1/CheckSeqLenIdentity$Predicates/CRF/rnn_1/sequence_length#^Predicates/CRF/rnn_1/Assert/Assert*#
_output_shapes
:?????????*
T0
j
Predicates/CRF/rnn_1/Shape_2ShapePredicates/CRF/rnn_1/transpose*
_output_shapes
:*
T0
t
*Predicates/CRF/rnn_1/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,Predicates/CRF/rnn_1/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,Predicates/CRF/rnn_1/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
$Predicates/CRF/rnn_1/strided_slice_1StridedSlicePredicates/CRF/rnn_1/Shape_2*Predicates/CRF/rnn_1/strided_slice_1/stack,Predicates/CRF/rnn_1/strided_slice_1/stack_1,Predicates/CRF/rnn_1/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0
j
Predicates/CRF/rnn_1/Shape_3ShapePredicates/CRF/rnn_1/transpose*
_output_shapes
:*
T0
t
*Predicates/CRF/rnn_1/strided_slice_2/stackConst*
valueB:*
_output_shapes
:*
dtype0
v
,Predicates/CRF/rnn_1/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
v
,Predicates/CRF/rnn_1/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
$Predicates/CRF/rnn_1/strided_slice_2StridedSlicePredicates/CRF/rnn_1/Shape_3*Predicates/CRF/rnn_1/strided_slice_2/stack,Predicates/CRF/rnn_1/strided_slice_2/stack_1,Predicates/CRF/rnn_1/strided_slice_2/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
e
#Predicates/CRF/rnn_1/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn_1/ExpandDims
ExpandDims$Predicates/CRF/rnn_1/strided_slice_2#Predicates/CRF/rnn_1/ExpandDims/dim*
T0*
_output_shapes
:
f
Predicates/CRF/rnn_1/Const_1Const*
dtype0*
valueB:*
_output_shapes
:
d
"Predicates/CRF/rnn_1/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn_1/concat_1ConcatV2Predicates/CRF/rnn_1/ExpandDimsPredicates/CRF/rnn_1/Const_1"Predicates/CRF/rnn_1/concat_1/axis*
N*
T0*
_output_shapes
:
b
 Predicates/CRF/rnn_1/zeros/ConstConst*
dtype0*
value	B : *
_output_shapes
: 
?
Predicates/CRF/rnn_1/zerosFillPredicates/CRF/rnn_1/concat_1 Predicates/CRF/rnn_1/zeros/Const*'
_output_shapes
:?????????*
T0
f
Predicates/CRF/rnn_1/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
?
Predicates/CRF/rnn_1/MinMin Predicates/CRF/rnn_1/CheckSeqLenPredicates/CRF/rnn_1/Const_2*
_output_shapes
: *
T0
f
Predicates/CRF/rnn_1/Const_3Const*
valueB: *
_output_shapes
:*
dtype0
?
Predicates/CRF/rnn_1/MaxMax Predicates/CRF/rnn_1/CheckSeqLenPredicates/CRF/rnn_1/Const_3*
_output_shapes
: *
T0
[
Predicates/CRF/rnn_1/timeConst*
dtype0*
value	B : *
_output_shapes
: 
?
 Predicates/CRF/rnn_1/TensorArrayTensorArrayV3$Predicates/CRF/rnn_1/strided_slice_1*
dtype0*$
element_shape:?????????*
_output_shapes

:: *
identical_element_shapes(*@
tensor_array_name+)Predicates/CRF/rnn_1/dynamic_rnn/output_0
?
"Predicates/CRF/rnn_1/TensorArray_1TensorArrayV3$Predicates/CRF/rnn_1/strided_slice_1*
dtype0*$
element_shape:?????????*
_output_shapes

:: *
identical_element_shapes(*?
tensor_array_name*(Predicates/CRF/rnn_1/dynamic_rnn/input_0
{
-Predicates/CRF/rnn_1/TensorArrayUnstack/ShapeShapePredicates/CRF/rnn_1/transpose*
_output_shapes
:*
T0
?
;Predicates/CRF/rnn_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
?
=Predicates/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
?
=Predicates/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
5Predicates/CRF/rnn_1/TensorArrayUnstack/strided_sliceStridedSlice-Predicates/CRF/rnn_1/TensorArrayUnstack/Shape;Predicates/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack=Predicates/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_1=Predicates/CRF/rnn_1/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
u
3Predicates/CRF/rnn_1/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
u
3Predicates/CRF/rnn_1/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
-Predicates/CRF/rnn_1/TensorArrayUnstack/rangeRange3Predicates/CRF/rnn_1/TensorArrayUnstack/range/start5Predicates/CRF/rnn_1/TensorArrayUnstack/strided_slice3Predicates/CRF/rnn_1/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????
?
OPredicates/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3"Predicates/CRF/rnn_1/TensorArray_1-Predicates/CRF/rnn_1/TensorArrayUnstack/rangePredicates/CRF/rnn_1/transpose$Predicates/CRF/rnn_1/TensorArray_1:1*1
_class'
%#loc:@Predicates/CRF/rnn_1/transpose*
T0*
_output_shapes
: 
`
Predicates/CRF/rnn_1/Maximum/xConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn_1/MaximumMaximumPredicates/CRF/rnn_1/Maximum/xPredicates/CRF/rnn_1/Max*
T0*
_output_shapes
: 
?
Predicates/CRF/rnn_1/MinimumMinimum$Predicates/CRF/rnn_1/strided_slice_1Predicates/CRF/rnn_1/Maximum*
T0*
_output_shapes
: 
n
,Predicates/CRF/rnn_1/while/iteration_counterConst*
value	B : *
_output_shapes
: *
dtype0
?
 Predicates/CRF/rnn_1/while/EnterEnter,Predicates/CRF/rnn_1/while/iteration_counter*8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
parallel_iterations *
T0*
_output_shapes
: 
?
"Predicates/CRF/rnn_1/while/Enter_1EnterPredicates/CRF/rnn_1/time*
_output_shapes
: *
parallel_iterations *8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
T0
?
"Predicates/CRF/rnn_1/while/Enter_2Enter"Predicates/CRF/rnn_1/TensorArray:1*
_output_shapes
: *
parallel_iterations *8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
T0
?
"Predicates/CRF/rnn_1/while/Enter_3EnterPredicates/CRF/ExpandDims_1*
parallel_iterations *'
_output_shapes
:?????????*8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
T0
?
 Predicates/CRF/rnn_1/while/MergeMerge Predicates/CRF/rnn_1/while/Enter(Predicates/CRF/rnn_1/while/NextIteration*
N*
_output_shapes
: : *
T0
?
"Predicates/CRF/rnn_1/while/Merge_1Merge"Predicates/CRF/rnn_1/while/Enter_1*Predicates/CRF/rnn_1/while/NextIteration_1*
N*
_output_shapes
: : *
T0
?
"Predicates/CRF/rnn_1/while/Merge_2Merge"Predicates/CRF/rnn_1/while/Enter_2*Predicates/CRF/rnn_1/while/NextIteration_2*
T0*
_output_shapes
: : *
N
?
"Predicates/CRF/rnn_1/while/Merge_3Merge"Predicates/CRF/rnn_1/while/Enter_3*Predicates/CRF/rnn_1/while/NextIteration_3*
N*)
_output_shapes
:?????????: *
T0
?
Predicates/CRF/rnn_1/while/LessLess Predicates/CRF/rnn_1/while/Merge%Predicates/CRF/rnn_1/while/Less/Enter*
_output_shapes
: *
T0
?
%Predicates/CRF/rnn_1/while/Less/EnterEnter$Predicates/CRF/rnn_1/strided_slice_1*
parallel_iterations *
is_constant(*
T0*
_output_shapes
: *8

frame_name*(Predicates/CRF/rnn_1/while/while_context
?
!Predicates/CRF/rnn_1/while/Less_1Less"Predicates/CRF/rnn_1/while/Merge_1'Predicates/CRF/rnn_1/while/Less_1/Enter*
T0*
_output_shapes
: 
?
'Predicates/CRF/rnn_1/while/Less_1/EnterEnterPredicates/CRF/rnn_1/Minimum*
parallel_iterations *
is_constant(*
T0*
_output_shapes
: *8

frame_name*(Predicates/CRF/rnn_1/while/while_context
?
%Predicates/CRF/rnn_1/while/LogicalAnd
LogicalAndPredicates/CRF/rnn_1/while/Less!Predicates/CRF/rnn_1/while/Less_1*
_output_shapes
: 
n
#Predicates/CRF/rnn_1/while/LoopCondLoopCond%Predicates/CRF/rnn_1/while/LogicalAnd*
_output_shapes
: 
?
!Predicates/CRF/rnn_1/while/SwitchSwitch Predicates/CRF/rnn_1/while/Merge#Predicates/CRF/rnn_1/while/LoopCond*
_output_shapes
: : *
T0*3
_class)
'%loc:@Predicates/CRF/rnn_1/while/Merge
?
#Predicates/CRF/rnn_1/while/Switch_1Switch"Predicates/CRF/rnn_1/while/Merge_1#Predicates/CRF/rnn_1/while/LoopCond*
T0*5
_class+
)'loc:@Predicates/CRF/rnn_1/while/Merge_1*
_output_shapes
: : 
?
#Predicates/CRF/rnn_1/while/Switch_2Switch"Predicates/CRF/rnn_1/while/Merge_2#Predicates/CRF/rnn_1/while/LoopCond*5
_class+
)'loc:@Predicates/CRF/rnn_1/while/Merge_2*
T0*
_output_shapes
: : 
?
#Predicates/CRF/rnn_1/while/Switch_3Switch"Predicates/CRF/rnn_1/while/Merge_3#Predicates/CRF/rnn_1/while/LoopCond*:
_output_shapes(
&:?????????:?????????*
T0*5
_class+
)'loc:@Predicates/CRF/rnn_1/while/Merge_3
u
#Predicates/CRF/rnn_1/while/IdentityIdentity#Predicates/CRF/rnn_1/while/Switch:1*
T0*
_output_shapes
: 
y
%Predicates/CRF/rnn_1/while/Identity_1Identity%Predicates/CRF/rnn_1/while/Switch_1:1*
_output_shapes
: *
T0
y
%Predicates/CRF/rnn_1/while/Identity_2Identity%Predicates/CRF/rnn_1/while/Switch_2:1*
_output_shapes
: *
T0
?
%Predicates/CRF/rnn_1/while/Identity_3Identity%Predicates/CRF/rnn_1/while/Switch_3:1*'
_output_shapes
:?????????*
T0
?
 Predicates/CRF/rnn_1/while/add/yConst$^Predicates/CRF/rnn_1/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn_1/while/addAdd#Predicates/CRF/rnn_1/while/Identity Predicates/CRF/rnn_1/while/add/y*
_output_shapes
: *
T0
?
,Predicates/CRF/rnn_1/while/TensorArrayReadV3TensorArrayReadV32Predicates/CRF/rnn_1/while/TensorArrayReadV3/Enter%Predicates/CRF/rnn_1/while/Identity_14Predicates/CRF/rnn_1/while/TensorArrayReadV3/Enter_1*'
_output_shapes
:?????????*
dtype0
?
2Predicates/CRF/rnn_1/while/TensorArrayReadV3/EnterEnter"Predicates/CRF/rnn_1/TensorArray_1*
is_constant(*
_output_shapes
:*
T0*8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
parallel_iterations 
?
4Predicates/CRF/rnn_1/while/TensorArrayReadV3/Enter_1EnterOPredicates/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *
T0*8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
parallel_iterations 
?
'Predicates/CRF/rnn_1/while/GreaterEqualGreaterEqual%Predicates/CRF/rnn_1/while/Identity_1-Predicates/CRF/rnn_1/while/GreaterEqual/Enter*#
_output_shapes
:?????????*
T0
?
-Predicates/CRF/rnn_1/while/GreaterEqual/EnterEnter Predicates/CRF/rnn_1/CheckSeqLen*8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
T0*#
_output_shapes
:?????????*
is_constant(*
parallel_iterations 
?
"Predicates/CRF/rnn_1/while/SqueezeSqueeze%Predicates/CRF/rnn_1/while/Identity_3*
T0*
squeeze_dims
*#
_output_shapes
:?????????
|
 Predicates/CRF/rnn_1/while/ShapeShape,Predicates/CRF/rnn_1/while/TensorArrayReadV3*
T0*
_output_shapes
:
?
.Predicates/CRF/rnn_1/while/strided_slice/stackConst$^Predicates/CRF/rnn_1/while/Identity*
valueB: *
_output_shapes
:*
dtype0
?
0Predicates/CRF/rnn_1/while/strided_slice/stack_1Const$^Predicates/CRF/rnn_1/while/Identity*
valueB:*
_output_shapes
:*
dtype0
?
0Predicates/CRF/rnn_1/while/strided_slice/stack_2Const$^Predicates/CRF/rnn_1/while/Identity*
_output_shapes
:*
dtype0*
valueB:
?
(Predicates/CRF/rnn_1/while/strided_sliceStridedSlice Predicates/CRF/rnn_1/while/Shape.Predicates/CRF/rnn_1/while/strided_slice/stack0Predicates/CRF/rnn_1/while/strided_slice/stack_10Predicates/CRF/rnn_1/while/strided_slice/stack_2*
shrink_axis_mask*
T0*
_output_shapes
: *
Index0
?
&Predicates/CRF/rnn_1/while/range/startConst$^Predicates/CRF/rnn_1/while/Identity*
value	B : *
_output_shapes
: *
dtype0
?
&Predicates/CRF/rnn_1/while/range/deltaConst$^Predicates/CRF/rnn_1/while/Identity*
value	B :*
_output_shapes
: *
dtype0
?
 Predicates/CRF/rnn_1/while/rangeRange&Predicates/CRF/rnn_1/while/range/start(Predicates/CRF/rnn_1/while/strided_slice&Predicates/CRF/rnn_1/while/range/delta*#
_output_shapes
:?????????
?
 Predicates/CRF/rnn_1/while/stackPack Predicates/CRF/rnn_1/while/range"Predicates/CRF/rnn_1/while/Squeeze*'
_output_shapes
:?????????*
T0*
N*

axis
?
#Predicates/CRF/rnn_1/while/GatherNdGatherNd,Predicates/CRF/rnn_1/while/TensorArrayReadV3 Predicates/CRF/rnn_1/while/stack*#
_output_shapes
:?????????*
Tparams0*
Tindices0
?
)Predicates/CRF/rnn_1/while/ExpandDims/dimConst$^Predicates/CRF/rnn_1/while/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????
?
%Predicates/CRF/rnn_1/while/ExpandDims
ExpandDims#Predicates/CRF/rnn_1/while/GatherNd)Predicates/CRF/rnn_1/while/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
!Predicates/CRF/rnn_1/while/SelectSelect'Predicates/CRF/rnn_1/while/GreaterEqual'Predicates/CRF/rnn_1/while/Select/Enter%Predicates/CRF/rnn_1/while/ExpandDims*'
_output_shapes
:?????????*
T0*8
_class.
,*loc:@Predicates/CRF/rnn_1/while/ExpandDims
?
'Predicates/CRF/rnn_1/while/Select/EnterEnterPredicates/CRF/rnn_1/zeros*'
_output_shapes
:?????????*
is_constant(*8

frame_name*(Predicates/CRF/rnn_1/while/while_context*
T0*8
_class.
,*loc:@Predicates/CRF/rnn_1/while/ExpandDims*
parallel_iterations 
?
#Predicates/CRF/rnn_1/while/Select_1Select'Predicates/CRF/rnn_1/while/GreaterEqual%Predicates/CRF/rnn_1/while/Identity_3%Predicates/CRF/rnn_1/while/ExpandDims*'
_output_shapes
:?????????*
T0*8
_class.
,*loc:@Predicates/CRF/rnn_1/while/ExpandDims
?
>Predicates/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3DPredicates/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter%Predicates/CRF/rnn_1/while/Identity_1!Predicates/CRF/rnn_1/while/Select%Predicates/CRF/rnn_1/while/Identity_2*
T0*
_output_shapes
: *8
_class.
,*loc:@Predicates/CRF/rnn_1/while/ExpandDims
?
DPredicates/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter Predicates/CRF/rnn_1/TensorArray*
T0*
parallel_iterations *
is_constant(*8

frame_name*(Predicates/CRF/rnn_1/while/while_context*8
_class.
,*loc:@Predicates/CRF/rnn_1/while/ExpandDims*
_output_shapes
:
?
"Predicates/CRF/rnn_1/while/add_1/yConst$^Predicates/CRF/rnn_1/while/Identity*
_output_shapes
: *
dtype0*
value	B :
?
 Predicates/CRF/rnn_1/while/add_1Add%Predicates/CRF/rnn_1/while/Identity_1"Predicates/CRF/rnn_1/while/add_1/y*
T0*
_output_shapes
: 
z
(Predicates/CRF/rnn_1/while/NextIterationNextIterationPredicates/CRF/rnn_1/while/add*
T0*
_output_shapes
: 
~
*Predicates/CRF/rnn_1/while/NextIteration_1NextIteration Predicates/CRF/rnn_1/while/add_1*
_output_shapes
: *
T0
?
*Predicates/CRF/rnn_1/while/NextIteration_2NextIteration>Predicates/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
?
*Predicates/CRF/rnn_1/while/NextIteration_3NextIteration#Predicates/CRF/rnn_1/while/Select_1*
T0*'
_output_shapes
:?????????
k
Predicates/CRF/rnn_1/while/ExitExit!Predicates/CRF/rnn_1/while/Switch*
T0*
_output_shapes
: 
o
!Predicates/CRF/rnn_1/while/Exit_1Exit#Predicates/CRF/rnn_1/while/Switch_1*
_output_shapes
: *
T0
o
!Predicates/CRF/rnn_1/while/Exit_2Exit#Predicates/CRF/rnn_1/while/Switch_2*
_output_shapes
: *
T0
?
!Predicates/CRF/rnn_1/while/Exit_3Exit#Predicates/CRF/rnn_1/while/Switch_3*
T0*'
_output_shapes
:?????????
?
7Predicates/CRF/rnn_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3 Predicates/CRF/rnn_1/TensorArray!Predicates/CRF/rnn_1/while/Exit_2*3
_class)
'%loc:@Predicates/CRF/rnn_1/TensorArray*
_output_shapes
: 
?
1Predicates/CRF/rnn_1/TensorArrayStack/range/startConst*3
_class)
'%loc:@Predicates/CRF/rnn_1/TensorArray*
_output_shapes
: *
dtype0*
value	B : 
?
1Predicates/CRF/rnn_1/TensorArrayStack/range/deltaConst*3
_class)
'%loc:@Predicates/CRF/rnn_1/TensorArray*
_output_shapes
: *
dtype0*
value	B :
?
+Predicates/CRF/rnn_1/TensorArrayStack/rangeRange1Predicates/CRF/rnn_1/TensorArrayStack/range/start7Predicates/CRF/rnn_1/TensorArrayStack/TensorArraySizeV31Predicates/CRF/rnn_1/TensorArrayStack/range/delta*3
_class)
'%loc:@Predicates/CRF/rnn_1/TensorArray*#
_output_shapes
:?????????
?
9Predicates/CRF/rnn_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3 Predicates/CRF/rnn_1/TensorArray+Predicates/CRF/rnn_1/TensorArrayStack/range!Predicates/CRF/rnn_1/while/Exit_2*3
_class)
'%loc:@Predicates/CRF/rnn_1/TensorArray*,
_output_shapes
:??????????*
dtype0*$
element_shape:?????????
f
Predicates/CRF/rnn_1/Const_4Const*
_output_shapes
:*
dtype0*
valueB:
]
Predicates/CRF/rnn_1/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
d
"Predicates/CRF/rnn_1/range_1/startConst*
value	B :*
_output_shapes
: *
dtype0
d
"Predicates/CRF/rnn_1/range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn_1/range_1Range"Predicates/CRF/rnn_1/range_1/startPredicates/CRF/rnn_1/Rank_1"Predicates/CRF/rnn_1/range_1/delta*
_output_shapes
:
w
&Predicates/CRF/rnn_1/concat_2/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
d
"Predicates/CRF/rnn_1/concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF/rnn_1/concat_2ConcatV2&Predicates/CRF/rnn_1/concat_2/values_0Predicates/CRF/rnn_1/range_1"Predicates/CRF/rnn_1/concat_2/axis*
N*
T0*
_output_shapes
:
?
 Predicates/CRF/rnn_1/transpose_1	Transpose9Predicates/CRF/rnn_1/TensorArrayStack/TensorArrayGatherV3Predicates/CRF/rnn_1/concat_2*
T0*,
_output_shapes
:??????????
?
Predicates/CRF/Squeeze_1Squeeze Predicates/CRF/rnn_1/transpose_1*
T0*(
_output_shapes
:??????????*
squeeze_dims

\
Predicates/CRF/concat/axisConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF/concatConcatV2Predicates/CRF/ExpandDims_1Predicates/CRF/Squeeze_1Predicates/CRF/concat/axis*
N*
T0*(
_output_shapes
:??????????
?
 Predicates/CRF/ReverseSequence_1ReverseSequencePredicates/CRF/concatpred_original_sequence_lengths*
T0*(
_output_shapes
:??????????*

Tlen0*
seq_dim
f
$Predicates/CRF/Max/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
?
Predicates/CRF/MaxMaxPredicates/CRF/rnn/while/Exit_3$Predicates/CRF/Max/reduction_indices*
T0*#
_output_shapes
:?????????
m
Predicates/Reshape/shapeConst*
dtype0*!
valueB"?????      *
_output_shapes
:
?
Predicates/ReshapeReshape Predicates/CRF/ReverseSequence_1Predicates/Reshape/shape*
T0*,
_output_shapes
:??????????
y
(Predicates/pred_arc/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
k
&Predicates/pred_arc/random_uniform/minConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
k
&Predicates/pred_arc/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
0Predicates/pred_arc/random_uniform/RandomUniformRandomUniform(Predicates/pred_arc/random_uniform/shape*
T0*
dtype0*
_output_shapes

:
?
&Predicates/pred_arc/random_uniform/subSub&Predicates/pred_arc/random_uniform/max&Predicates/pred_arc/random_uniform/min*
T0*
_output_shapes
: 
?
&Predicates/pred_arc/random_uniform/mulMul0Predicates/pred_arc/random_uniform/RandomUniform&Predicates/pred_arc/random_uniform/sub*
T0*
_output_shapes

:
?
"Predicates/pred_arc/random_uniformAdd&Predicates/pred_arc/random_uniform/mul&Predicates/pred_arc/random_uniform/min*
T0*
_output_shapes

:
l
Predicates/pred_arc/Variable
VariableV2*
dtype0*
shape
:*
_output_shapes

:
?
#Predicates/pred_arc/Variable/AssignAssignPredicates/pred_arc/Variable"Predicates/pred_arc/random_uniform*
T0*/
_class%
#!loc:@Predicates/pred_arc/Variable*
_output_shapes

:
?
!Predicates/pred_arc/Variable/readIdentityPredicates/pred_arc/Variable*
T0*/
_class%
#!loc:@Predicates/pred_arc/Variable*
_output_shapes

:
?
)Predicates/pred_arc/embedding_lookup/axisConst*
dtype0*
value	B : */
_class%
#!loc:@Predicates/pred_arc/Variable*
_output_shapes
: 
?
$Predicates/pred_arc/embedding_lookupGatherV2!Predicates/pred_arc/Variable/readPredicates/Reshape)Predicates/pred_arc/embedding_lookup/axis*
Taxis0*/
_class%
#!loc:@Predicates/pred_arc/Variable*
Tindices0*
Tparams0*0
_output_shapes
:??????????
?
-Predicates/pred_arc/embedding_lookup/IdentityIdentity$Predicates/pred_arc/embedding_lookup*
T0*0
_output_shapes
:??????????
?
Predicates/pred_arc/SqueezeSqueeze-Predicates/pred_arc/embedding_lookup/Identity*
squeeze_dims
*
T0*,
_output_shapes
:??????????
c
Predicates/concat_3/axisConst*
dtype0*
valueB :
?????????*
_output_shapes
: 
?
Predicates/concat_3ConcatV2Predicates/MeanPredicates/pred_arc/SqueezePredicates/concat_3/axis*
T0*-
_output_shapes
:???????????*
N
o
Predicates/Reshape_1/shapeConst*
dtype0*!
valueB"?????   ?   *
_output_shapes
:
?
Predicates/Reshape_1ReshapePredicates/concat_3Predicates/Reshape_1/shape*
T0*-
_output_shapes
:???????????
?
5Predicates/proj_2/W_2/Initializer/random_normal/shapeConst*(
_class
loc:@Predicates/proj_2/W_2*
dtype0*
valueB"?      *
_output_shapes
:
?
4Predicates/proj_2/W_2/Initializer/random_normal/meanConst*(
_class
loc:@Predicates/proj_2/W_2*
dtype0*
valueB
 *    *
_output_shapes
: 
?
6Predicates/proj_2/W_2/Initializer/random_normal/stddevConst*(
_class
loc:@Predicates/proj_2/W_2*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
DPredicates/proj_2/W_2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal5Predicates/proj_2/W_2/Initializer/random_normal/shape*
T0*(
_class
loc:@Predicates/proj_2/W_2*
dtype0*
_output_shapes
:	?
?
3Predicates/proj_2/W_2/Initializer/random_normal/mulMulDPredicates/proj_2/W_2/Initializer/random_normal/RandomStandardNormal6Predicates/proj_2/W_2/Initializer/random_normal/stddev*
T0*(
_class
loc:@Predicates/proj_2/W_2*
_output_shapes
:	?
?
/Predicates/proj_2/W_2/Initializer/random_normalAdd3Predicates/proj_2/W_2/Initializer/random_normal/mul4Predicates/proj_2/W_2/Initializer/random_normal/mean*
T0*(
_class
loc:@Predicates/proj_2/W_2*
_output_shapes
:	?
?
$Predicates/proj_2/W_2/Initializer/QrQr/Predicates/proj_2/W_2/Initializer/random_normal*)
_output_shapes
:	?:*(
_class
loc:@Predicates/proj_2/W_2*
T0
?
*Predicates/proj_2/W_2/Initializer/DiagPartDiagPart&Predicates/proj_2/W_2/Initializer/Qr:1*
_output_shapes
:*(
_class
loc:@Predicates/proj_2/W_2*
T0
?
&Predicates/proj_2/W_2/Initializer/SignSign*Predicates/proj_2/W_2/Initializer/DiagPart*
_output_shapes
:*(
_class
loc:@Predicates/proj_2/W_2*
T0
?
%Predicates/proj_2/W_2/Initializer/mulMul$Predicates/proj_2/W_2/Initializer/Qr&Predicates/proj_2/W_2/Initializer/Sign*
_output_shapes
:	?*(
_class
loc:@Predicates/proj_2/W_2*
T0
?
/Predicates/proj_2/W_2/Initializer/Reshape/shapeConst*
_output_shapes
:*
dtype0*(
_class
loc:@Predicates/proj_2/W_2*
valueB"?      
?
)Predicates/proj_2/W_2/Initializer/ReshapeReshape%Predicates/proj_2/W_2/Initializer/mul/Predicates/proj_2/W_2/Initializer/Reshape/shape*
_output_shapes
:	?*
T0*(
_class
loc:@Predicates/proj_2/W_2
?
)Predicates/proj_2/W_2/Initializer/mul_1/xConst*
_output_shapes
: *
dtype0*(
_class
loc:@Predicates/proj_2/W_2*
valueB
 *???
?
'Predicates/proj_2/W_2/Initializer/mul_1Mul)Predicates/proj_2/W_2/Initializer/mul_1/x)Predicates/proj_2/W_2/Initializer/Reshape*
_output_shapes
:	?*(
_class
loc:@Predicates/proj_2/W_2*
T0
?
Predicates/proj_2/W_2
VariableV2*(
_class
loc:@Predicates/proj_2/W_2*
shape:	?*
_output_shapes
:	?*
dtype0
?
Predicates/proj_2/W_2/AssignAssignPredicates/proj_2/W_2'Predicates/proj_2/W_2/Initializer/mul_1*
_output_shapes
:	?*
T0*(
_class
loc:@Predicates/proj_2/W_2
?
Predicates/proj_2/W_2/readIdentityPredicates/proj_2/W_2*(
_class
loc:@Predicates/proj_2/W_2*
T0*
_output_shapes
:	?
?
'Predicates/proj_2/b_2/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*(
_class
loc:@Predicates/proj_2/b_2
?
Predicates/proj_2/b_2
VariableV2*
dtype0*
shape:*
_output_shapes
:*(
_class
loc:@Predicates/proj_2/b_2
?
Predicates/proj_2/b_2/AssignAssignPredicates/proj_2/b_2'Predicates/proj_2/b_2/Initializer/zeros*
_output_shapes
:*
T0*(
_class
loc:@Predicates/proj_2/b_2
?
Predicates/proj_2/b_2/readIdentityPredicates/proj_2/b_2*
_output_shapes
:*
T0*(
_class
loc:@Predicates/proj_2/b_2
p
Predicates/proj_2/Reshape/shapeConst*
dtype0*
valueB"?????   *
_output_shapes
:
?
Predicates/proj_2/ReshapeReshapePredicates/Reshape_1Predicates/proj_2/Reshape/shape*(
_output_shapes
:??????????*
T0
?
Predicates/proj_2/MatMulMatMulPredicates/proj_2/ReshapePredicates/proj_2/W_2/read*
T0*'
_output_shapes
:?????????
?
Predicates/proj_2/addAddPredicates/proj_2/MatMulPredicates/proj_2/b_2/read*
T0*'
_output_shapes
:?????????
v
!Predicates/proj_2/Reshape_1/shapeConst*!
valueB"?????      *
_output_shapes
:*
dtype0
?
Predicates/proj_2/Reshape_1ReshapePredicates/proj_2/add!Predicates/proj_2/Reshape_1/shape*,
_output_shapes
:??????????*
T0
?
>Predicates/CRF_2/transition_2/Initializer/random_uniform/shapeConst*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
valueB"      *
_output_shapes
:*
dtype0
?
<Predicates/CRF_2/transition_2/Initializer/random_uniform/minConst*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
valueB
 *qĜ?*
_output_shapes
: *
dtype0
?
<Predicates/CRF_2/transition_2/Initializer/random_uniform/maxConst*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
valueB
 *qĜ?*
_output_shapes
: *
dtype0
?
FPredicates/CRF_2/transition_2/Initializer/random_uniform/RandomUniformRandomUniform>Predicates/CRF_2/transition_2/Initializer/random_uniform/shape*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
T0*
_output_shapes

:*
dtype0
?
<Predicates/CRF_2/transition_2/Initializer/random_uniform/subSub<Predicates/CRF_2/transition_2/Initializer/random_uniform/max<Predicates/CRF_2/transition_2/Initializer/random_uniform/min*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
T0*
_output_shapes
: 
?
<Predicates/CRF_2/transition_2/Initializer/random_uniform/mulMulFPredicates/CRF_2/transition_2/Initializer/random_uniform/RandomUniform<Predicates/CRF_2/transition_2/Initializer/random_uniform/sub*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
T0*
_output_shapes

:
?
8Predicates/CRF_2/transition_2/Initializer/random_uniformAdd<Predicates/CRF_2/transition_2/Initializer/random_uniform/mul<Predicates/CRF_2/transition_2/Initializer/random_uniform/min*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
_output_shapes

:*
T0
?
Predicates/CRF_2/transition_2
VariableV2*
dtype0*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
_output_shapes

:*
shape
:
?
$Predicates/CRF_2/transition_2/AssignAssignPredicates/CRF_2/transition_28Predicates/CRF_2/transition_2/Initializer/random_uniform*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
_output_shapes

:*
T0
?
"Predicates/CRF_2/transition_2/readIdentityPredicates/CRF_2/transition_2*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
_output_shapes

:*
T0
[
Predicates/CRF_2/Equal/xConst*
dtype0*
value
B :?*
_output_shapes
: 
Z
Predicates/CRF_2/Equal/yConst*
value	B :*
_output_shapes
: *
dtype0
t
Predicates/CRF_2/EqualEqualPredicates/CRF_2/Equal/xPredicates/CRF_2/Equal/y*
T0*
_output_shapes
: 
a
Predicates/CRF_2/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
Predicates/CRF_2/ExpandDims
ExpandDims"Predicates/CRF_2/transition_2/readPredicates/CRF_2/ExpandDims/dim*"
_output_shapes
:*
T0
q
Predicates/CRF_2/Slice/beginConst*
dtype0*!
valueB"            *
_output_shapes
:
p
Predicates/CRF_2/Slice/sizeConst*
dtype0*!
valueB"????   ????*
_output_shapes
:
?
Predicates/CRF_2/SliceSlicePredicates/proj_2/Reshape_1Predicates/CRF_2/Slice/beginPredicates/CRF_2/Slice/size*
Index0*
T0*+
_output_shapes
:?????????
?
Predicates/CRF_2/SqueezeSqueezePredicates/CRF_2/Slice*
squeeze_dims
*'
_output_shapes
:?????????*
T0
s
Predicates/CRF_2/Slice_1/beginConst*!
valueB"           *
_output_shapes
:*
dtype0
r
Predicates/CRF_2/Slice_1/sizeConst*!
valueB"????????????*
_output_shapes
:*
dtype0
?
Predicates/CRF_2/Slice_1SlicePredicates/proj_2/Reshape_1Predicates/CRF_2/Slice_1/beginPredicates/CRF_2/Slice_1/size*
Index0*
T0*,
_output_shapes
:??????????
X
Predicates/CRF_2/ConstConst*
value	B : *
_output_shapes
: *
dtype0
X
Predicates/CRF_2/sub/yConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF_2/subSubpred_original_sequence_lengthsPredicates/CRF_2/sub/y*
T0*#
_output_shapes
:?????????

Predicates/CRF_2/MaximumMaximumPredicates/CRF_2/ConstPredicates/CRF_2/sub*
T0*#
_output_shapes
:?????????
[
Predicates/CRF_2/rnn/RankConst*
value	B :*
_output_shapes
: *
dtype0
b
 Predicates/CRF_2/rnn/range/startConst*
value	B :*
_output_shapes
: *
dtype0
b
 Predicates/CRF_2/rnn/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn/rangeRange Predicates/CRF_2/rnn/range/startPredicates/CRF_2/rnn/Rank Predicates/CRF_2/rnn/range/delta*
_output_shapes
:
u
$Predicates/CRF_2/rnn/concat/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
b
 Predicates/CRF_2/rnn/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn/concatConcatV2$Predicates/CRF_2/rnn/concat/values_0Predicates/CRF_2/rnn/range Predicates/CRF_2/rnn/concat/axis*
N*
_output_shapes
:*
T0
?
Predicates/CRF_2/rnn/transpose	TransposePredicates/CRF_2/Slice_1Predicates/CRF_2/rnn/concat*,
_output_shapes
:??????????*
T0
x
$Predicates/CRF_2/rnn/sequence_lengthIdentityPredicates/CRF_2/Maximum*#
_output_shapes
:?????????*
T0
h
Predicates/CRF_2/rnn/ShapeShapePredicates/CRF_2/rnn/transpose*
_output_shapes
:*
T0
r
(Predicates/CRF_2/rnn/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
t
*Predicates/CRF_2/rnn/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
t
*Predicates/CRF_2/rnn/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
"Predicates/CRF_2/rnn/strided_sliceStridedSlicePredicates/CRF_2/rnn/Shape(Predicates/CRF_2/rnn/strided_slice/stack*Predicates/CRF_2/rnn/strided_slice/stack_1*Predicates/CRF_2/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
p
Predicates/CRF_2/rnn/Shape_1Shape$Predicates/CRF_2/rnn/sequence_length*
T0*
_output_shapes
:
t
Predicates/CRF_2/rnn/stackPack"Predicates/CRF_2/rnn/strided_slice*
T0*
N*
_output_shapes
:
?
Predicates/CRF_2/rnn/EqualEqualPredicates/CRF_2/rnn/Shape_1Predicates/CRF_2/rnn/stack*
T0*
_output_shapes
:
d
Predicates/CRF_2/rnn/ConstConst*
dtype0*
valueB: *
_output_shapes
:
o
Predicates/CRF_2/rnn/AllAllPredicates/CRF_2/rnn/EqualPredicates/CRF_2/rnn/Const*
_output_shapes
: 
?
!Predicates/CRF_2/rnn/Assert/ConstConst*
dtype0*U
valueLBJ BDExpected shape for Tensor Predicates/CRF_2/rnn/sequence_length:0 is *
_output_shapes
: 
t
#Predicates/CRF_2/rnn/Assert/Const_1Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
?
)Predicates/CRF_2/rnn/Assert/Assert/data_0Const*
dtype0*U
valueLBJ BDExpected shape for Tensor Predicates/CRF_2/rnn/sequence_length:0 is *
_output_shapes
: 
z
)Predicates/CRF_2/rnn/Assert/Assert/data_2Const*
dtype0*!
valueB B but saw shape: *
_output_shapes
: 
?
"Predicates/CRF_2/rnn/Assert/AssertAssertPredicates/CRF_2/rnn/All)Predicates/CRF_2/rnn/Assert/Assert/data_0Predicates/CRF_2/rnn/stack)Predicates/CRF_2/rnn/Assert/Assert/data_2Predicates/CRF_2/rnn/Shape_1*
T
2
?
 Predicates/CRF_2/rnn/CheckSeqLenIdentity$Predicates/CRF_2/rnn/sequence_length#^Predicates/CRF_2/rnn/Assert/Assert*
T0*#
_output_shapes
:?????????
j
Predicates/CRF_2/rnn/Shape_2ShapePredicates/CRF_2/rnn/transpose*
T0*
_output_shapes
:
t
*Predicates/CRF_2/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
v
,Predicates/CRF_2/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
v
,Predicates/CRF_2/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
$Predicates/CRF_2/rnn/strided_slice_1StridedSlicePredicates/CRF_2/rnn/Shape_2*Predicates/CRF_2/rnn/strided_slice_1/stack,Predicates/CRF_2/rnn/strided_slice_1/stack_1,Predicates/CRF_2/rnn/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
j
Predicates/CRF_2/rnn/Shape_3ShapePredicates/CRF_2/rnn/transpose*
T0*
_output_shapes
:
t
*Predicates/CRF_2/rnn/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0
v
,Predicates/CRF_2/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
v
,Predicates/CRF_2/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
$Predicates/CRF_2/rnn/strided_slice_2StridedSlicePredicates/CRF_2/rnn/Shape_3*Predicates/CRF_2/rnn/strided_slice_2/stack,Predicates/CRF_2/rnn/strided_slice_2/stack_1,Predicates/CRF_2/rnn/strided_slice_2/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
e
#Predicates/CRF_2/rnn/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn/ExpandDims
ExpandDims$Predicates/CRF_2/rnn/strided_slice_2#Predicates/CRF_2/rnn/ExpandDims/dim*
_output_shapes
:*
T0
f
Predicates/CRF_2/rnn/Const_1Const*
valueB:*
_output_shapes
:*
dtype0
d
"Predicates/CRF_2/rnn/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn/concat_1ConcatV2Predicates/CRF_2/rnn/ExpandDimsPredicates/CRF_2/rnn/Const_1"Predicates/CRF_2/rnn/concat_1/axis*
_output_shapes
:*
T0*
N
b
 Predicates/CRF_2/rnn/zeros/ConstConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn/zerosFillPredicates/CRF_2/rnn/concat_1 Predicates/CRF_2/rnn/zeros/Const*
T0*'
_output_shapes
:?????????
f
Predicates/CRF_2/rnn/Const_2Const*
valueB: *
_output_shapes
:*
dtype0
?
Predicates/CRF_2/rnn/MinMin Predicates/CRF_2/rnn/CheckSeqLenPredicates/CRF_2/rnn/Const_2*
T0*
_output_shapes
: 
f
Predicates/CRF_2/rnn/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
?
Predicates/CRF_2/rnn/MaxMax Predicates/CRF_2/rnn/CheckSeqLenPredicates/CRF_2/rnn/Const_3*
T0*
_output_shapes
: 
[
Predicates/CRF_2/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
?
 Predicates/CRF_2/rnn/TensorArrayTensorArrayV3$Predicates/CRF_2/rnn/strided_slice_1*
dtype0*$
element_shape:?????????*
_output_shapes

:: *@
tensor_array_name+)Predicates/CRF_2/rnn/dynamic_rnn/output_0*
identical_element_shapes(
?
"Predicates/CRF_2/rnn/TensorArray_1TensorArrayV3$Predicates/CRF_2/rnn/strided_slice_1*
dtype0*$
element_shape:?????????*
_output_shapes

:: *?
tensor_array_name*(Predicates/CRF_2/rnn/dynamic_rnn/input_0*
identical_element_shapes(
{
-Predicates/CRF_2/rnn/TensorArrayUnstack/ShapeShapePredicates/CRF_2/rnn/transpose*
T0*
_output_shapes
:
?
;Predicates/CRF_2/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
?
=Predicates/CRF_2/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
?
=Predicates/CRF_2/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
5Predicates/CRF_2/rnn/TensorArrayUnstack/strided_sliceStridedSlice-Predicates/CRF_2/rnn/TensorArrayUnstack/Shape;Predicates/CRF_2/rnn/TensorArrayUnstack/strided_slice/stack=Predicates/CRF_2/rnn/TensorArrayUnstack/strided_slice/stack_1=Predicates/CRF_2/rnn/TensorArrayUnstack/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
u
3Predicates/CRF_2/rnn/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
u
3Predicates/CRF_2/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
-Predicates/CRF_2/rnn/TensorArrayUnstack/rangeRange3Predicates/CRF_2/rnn/TensorArrayUnstack/range/start5Predicates/CRF_2/rnn/TensorArrayUnstack/strided_slice3Predicates/CRF_2/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????
?
OPredicates/CRF_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3"Predicates/CRF_2/rnn/TensorArray_1-Predicates/CRF_2/rnn/TensorArrayUnstack/rangePredicates/CRF_2/rnn/transpose$Predicates/CRF_2/rnn/TensorArray_1:1*1
_class'
%#loc:@Predicates/CRF_2/rnn/transpose*
_output_shapes
: *
T0
`
Predicates/CRF_2/rnn/Maximum/xConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn/MaximumMaximumPredicates/CRF_2/rnn/Maximum/xPredicates/CRF_2/rnn/Max*
_output_shapes
: *
T0
?
Predicates/CRF_2/rnn/MinimumMinimum$Predicates/CRF_2/rnn/strided_slice_1Predicates/CRF_2/rnn/Maximum*
_output_shapes
: *
T0
n
,Predicates/CRF_2/rnn/while/iteration_counterConst*
value	B : *
_output_shapes
: *
dtype0
?
 Predicates/CRF_2/rnn/while/EnterEnter,Predicates/CRF_2/rnn/while/iteration_counter*
parallel_iterations *
T0*
_output_shapes
: *8

frame_name*(Predicates/CRF_2/rnn/while/while_context
?
"Predicates/CRF_2/rnn/while/Enter_1EnterPredicates/CRF_2/rnn/time*
parallel_iterations *
T0*
_output_shapes
: *8

frame_name*(Predicates/CRF_2/rnn/while/while_context
?
"Predicates/CRF_2/rnn/while/Enter_2Enter"Predicates/CRF_2/rnn/TensorArray:1*
parallel_iterations *
T0*
_output_shapes
: *8

frame_name*(Predicates/CRF_2/rnn/while/while_context
?
"Predicates/CRF_2/rnn/while/Enter_3EnterPredicates/CRF_2/Squeeze*
parallel_iterations *
T0*'
_output_shapes
:?????????*8

frame_name*(Predicates/CRF_2/rnn/while/while_context
?
 Predicates/CRF_2/rnn/while/MergeMerge Predicates/CRF_2/rnn/while/Enter(Predicates/CRF_2/rnn/while/NextIteration*
T0*
_output_shapes
: : *
N
?
"Predicates/CRF_2/rnn/while/Merge_1Merge"Predicates/CRF_2/rnn/while/Enter_1*Predicates/CRF_2/rnn/while/NextIteration_1*
T0*
_output_shapes
: : *
N
?
"Predicates/CRF_2/rnn/while/Merge_2Merge"Predicates/CRF_2/rnn/while/Enter_2*Predicates/CRF_2/rnn/while/NextIteration_2*
T0*
_output_shapes
: : *
N
?
"Predicates/CRF_2/rnn/while/Merge_3Merge"Predicates/CRF_2/rnn/while/Enter_3*Predicates/CRF_2/rnn/while/NextIteration_3*
T0*)
_output_shapes
:?????????: *
N
?
Predicates/CRF_2/rnn/while/LessLess Predicates/CRF_2/rnn/while/Merge%Predicates/CRF_2/rnn/while/Less/Enter*
T0*
_output_shapes
: 
?
%Predicates/CRF_2/rnn/while/Less/EnterEnter$Predicates/CRF_2/rnn/strided_slice_1*8

frame_name*(Predicates/CRF_2/rnn/while/while_context*
T0*
_output_shapes
: *
is_constant(*
parallel_iterations 
?
!Predicates/CRF_2/rnn/while/Less_1Less"Predicates/CRF_2/rnn/while/Merge_1'Predicates/CRF_2/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
?
'Predicates/CRF_2/rnn/while/Less_1/EnterEnterPredicates/CRF_2/rnn/Minimum*
_output_shapes
: *
is_constant(*
T0*
parallel_iterations *8

frame_name*(Predicates/CRF_2/rnn/while/while_context
?
%Predicates/CRF_2/rnn/while/LogicalAnd
LogicalAndPredicates/CRF_2/rnn/while/Less!Predicates/CRF_2/rnn/while/Less_1*
_output_shapes
: 
n
#Predicates/CRF_2/rnn/while/LoopCondLoopCond%Predicates/CRF_2/rnn/while/LogicalAnd*
_output_shapes
: 
?
!Predicates/CRF_2/rnn/while/SwitchSwitch Predicates/CRF_2/rnn/while/Merge#Predicates/CRF_2/rnn/while/LoopCond*
_output_shapes
: : *
T0*3
_class)
'%loc:@Predicates/CRF_2/rnn/while/Merge
?
#Predicates/CRF_2/rnn/while/Switch_1Switch"Predicates/CRF_2/rnn/while/Merge_1#Predicates/CRF_2/rnn/while/LoopCond*
_output_shapes
: : *
T0*5
_class+
)'loc:@Predicates/CRF_2/rnn/while/Merge_1
?
#Predicates/CRF_2/rnn/while/Switch_2Switch"Predicates/CRF_2/rnn/while/Merge_2#Predicates/CRF_2/rnn/while/LoopCond*
_output_shapes
: : *
T0*5
_class+
)'loc:@Predicates/CRF_2/rnn/while/Merge_2
?
#Predicates/CRF_2/rnn/while/Switch_3Switch"Predicates/CRF_2/rnn/while/Merge_3#Predicates/CRF_2/rnn/while/LoopCond*:
_output_shapes(
&:?????????:?????????*
T0*5
_class+
)'loc:@Predicates/CRF_2/rnn/while/Merge_3
u
#Predicates/CRF_2/rnn/while/IdentityIdentity#Predicates/CRF_2/rnn/while/Switch:1*
T0*
_output_shapes
: 
y
%Predicates/CRF_2/rnn/while/Identity_1Identity%Predicates/CRF_2/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
y
%Predicates/CRF_2/rnn/while/Identity_2Identity%Predicates/CRF_2/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
?
%Predicates/CRF_2/rnn/while/Identity_3Identity%Predicates/CRF_2/rnn/while/Switch_3:1*
T0*'
_output_shapes
:?????????
?
 Predicates/CRF_2/rnn/while/add/yConst$^Predicates/CRF_2/rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
?
Predicates/CRF_2/rnn/while/addAdd#Predicates/CRF_2/rnn/while/Identity Predicates/CRF_2/rnn/while/add/y*
T0*
_output_shapes
: 
?
,Predicates/CRF_2/rnn/while/TensorArrayReadV3TensorArrayReadV32Predicates/CRF_2/rnn/while/TensorArrayReadV3/Enter%Predicates/CRF_2/rnn/while/Identity_14Predicates/CRF_2/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:?????????
?
2Predicates/CRF_2/rnn/while/TensorArrayReadV3/EnterEnter"Predicates/CRF_2/rnn/TensorArray_1*
_output_shapes
:*8

frame_name*(Predicates/CRF_2/rnn/while/while_context*
parallel_iterations *
T0*
is_constant(
?
4Predicates/CRF_2/rnn/while/TensorArrayReadV3/Enter_1EnterOPredicates/CRF_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*8

frame_name*(Predicates/CRF_2/rnn/while/while_context*
T0*
_output_shapes
: *
is_constant(*
parallel_iterations 
?
'Predicates/CRF_2/rnn/while/GreaterEqualGreaterEqual%Predicates/CRF_2/rnn/while/Identity_1-Predicates/CRF_2/rnn/while/GreaterEqual/Enter*
T0*#
_output_shapes
:?????????
?
-Predicates/CRF_2/rnn/while/GreaterEqual/EnterEnter Predicates/CRF_2/rnn/CheckSeqLen*#
_output_shapes
:?????????*8

frame_name*(Predicates/CRF_2/rnn/while/while_context*
parallel_iterations *
T0*
is_constant(
?
)Predicates/CRF_2/rnn/while/ExpandDims/dimConst$^Predicates/CRF_2/rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
?
%Predicates/CRF_2/rnn/while/ExpandDims
ExpandDims%Predicates/CRF_2/rnn/while/Identity_3)Predicates/CRF_2/rnn/while/ExpandDims/dim*+
_output_shapes
:?????????*
T0
?
 Predicates/CRF_2/rnn/while/add_1Add%Predicates/CRF_2/rnn/while/ExpandDims&Predicates/CRF_2/rnn/while/add_1/Enter*+
_output_shapes
:?????????*
T0
?
&Predicates/CRF_2/rnn/while/add_1/EnterEnterPredicates/CRF_2/ExpandDims*"
_output_shapes
:*8

frame_name*(Predicates/CRF_2/rnn/while/while_context*
parallel_iterations *
T0*
is_constant(
?
0Predicates/CRF_2/rnn/while/Max/reduction_indicesConst$^Predicates/CRF_2/rnn/while/Identity*
_output_shapes
:*
dtype0*
valueB:
?
Predicates/CRF_2/rnn/while/MaxMax Predicates/CRF_2/rnn/while/add_10Predicates/CRF_2/rnn/while/Max/reduction_indices*'
_output_shapes
:?????????*
T0
?
 Predicates/CRF_2/rnn/while/add_2Add,Predicates/CRF_2/rnn/while/TensorArrayReadV3Predicates/CRF_2/rnn/while/Max*
T0*'
_output_shapes
:?????????
?
+Predicates/CRF_2/rnn/while/ArgMax/dimensionConst$^Predicates/CRF_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
?
!Predicates/CRF_2/rnn/while/ArgMaxArgMax Predicates/CRF_2/rnn/while/add_1+Predicates/CRF_2/rnn/while/ArgMax/dimension*
T0*'
_output_shapes
:?????????
?
Predicates/CRF_2/rnn/while/CastCast!Predicates/CRF_2/rnn/while/ArgMax*'
_output_shapes
:?????????*

DstT0*

SrcT0	
?
!Predicates/CRF_2/rnn/while/SelectSelect'Predicates/CRF_2/rnn/while/GreaterEqual'Predicates/CRF_2/rnn/while/Select/EnterPredicates/CRF_2/rnn/while/Cast*
T0*'
_output_shapes
:?????????*2
_class(
&$loc:@Predicates/CRF_2/rnn/while/Cast
?
'Predicates/CRF_2/rnn/while/Select/EnterEnterPredicates/CRF_2/rnn/zeros*
is_constant(*
parallel_iterations *'
_output_shapes
:?????????*2
_class(
&$loc:@Predicates/CRF_2/rnn/while/Cast*8

frame_name*(Predicates/CRF_2/rnn/while/while_context*
T0
?
#Predicates/CRF_2/rnn/while/Select_1Select'Predicates/CRF_2/rnn/while/GreaterEqual%Predicates/CRF_2/rnn/while/Identity_3 Predicates/CRF_2/rnn/while/add_2*
T0*'
_output_shapes
:?????????*3
_class)
'%loc:@Predicates/CRF_2/rnn/while/add_2
?
>Predicates/CRF_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3DPredicates/CRF_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter%Predicates/CRF_2/rnn/while/Identity_1!Predicates/CRF_2/rnn/while/Select%Predicates/CRF_2/rnn/while/Identity_2*
T0*
_output_shapes
: *2
_class(
&$loc:@Predicates/CRF_2/rnn/while/Cast
?
DPredicates/CRF_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter Predicates/CRF_2/rnn/TensorArray*
T0*
is_constant(*
_output_shapes
:*2
_class(
&$loc:@Predicates/CRF_2/rnn/while/Cast*
parallel_iterations *8

frame_name*(Predicates/CRF_2/rnn/while/while_context
?
"Predicates/CRF_2/rnn/while/add_3/yConst$^Predicates/CRF_2/rnn/while/Identity*
dtype0*
value	B :*
_output_shapes
: 
?
 Predicates/CRF_2/rnn/while/add_3Add%Predicates/CRF_2/rnn/while/Identity_1"Predicates/CRF_2/rnn/while/add_3/y*
T0*
_output_shapes
: 
z
(Predicates/CRF_2/rnn/while/NextIterationNextIterationPredicates/CRF_2/rnn/while/add*
T0*
_output_shapes
: 
~
*Predicates/CRF_2/rnn/while/NextIteration_1NextIteration Predicates/CRF_2/rnn/while/add_3*
T0*
_output_shapes
: 
?
*Predicates/CRF_2/rnn/while/NextIteration_2NextIteration>Predicates/CRF_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
?
*Predicates/CRF_2/rnn/while/NextIteration_3NextIteration#Predicates/CRF_2/rnn/while/Select_1*
T0*'
_output_shapes
:?????????
k
Predicates/CRF_2/rnn/while/ExitExit!Predicates/CRF_2/rnn/while/Switch*
T0*
_output_shapes
: 
o
!Predicates/CRF_2/rnn/while/Exit_1Exit#Predicates/CRF_2/rnn/while/Switch_1*
_output_shapes
: *
T0
o
!Predicates/CRF_2/rnn/while/Exit_2Exit#Predicates/CRF_2/rnn/while/Switch_2*
T0*
_output_shapes
: 
?
!Predicates/CRF_2/rnn/while/Exit_3Exit#Predicates/CRF_2/rnn/while/Switch_3*'
_output_shapes
:?????????*
T0
?
7Predicates/CRF_2/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3 Predicates/CRF_2/rnn/TensorArray!Predicates/CRF_2/rnn/while/Exit_2*3
_class)
'%loc:@Predicates/CRF_2/rnn/TensorArray*
_output_shapes
: 
?
1Predicates/CRF_2/rnn/TensorArrayStack/range/startConst*
dtype0*3
_class)
'%loc:@Predicates/CRF_2/rnn/TensorArray*
value	B : *
_output_shapes
: 
?
1Predicates/CRF_2/rnn/TensorArrayStack/range/deltaConst*3
_class)
'%loc:@Predicates/CRF_2/rnn/TensorArray*
_output_shapes
: *
value	B :*
dtype0
?
+Predicates/CRF_2/rnn/TensorArrayStack/rangeRange1Predicates/CRF_2/rnn/TensorArrayStack/range/start7Predicates/CRF_2/rnn/TensorArrayStack/TensorArraySizeV31Predicates/CRF_2/rnn/TensorArrayStack/range/delta*3
_class)
'%loc:@Predicates/CRF_2/rnn/TensorArray*#
_output_shapes
:?????????
?
9Predicates/CRF_2/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3 Predicates/CRF_2/rnn/TensorArray+Predicates/CRF_2/rnn/TensorArrayStack/range!Predicates/CRF_2/rnn/while/Exit_2*3
_class)
'%loc:@Predicates/CRF_2/rnn/TensorArray*,
_output_shapes
:??????????*$
element_shape:?????????*
dtype0
f
Predicates/CRF_2/rnn/Const_4Const*
_output_shapes
:*
valueB:*
dtype0
]
Predicates/CRF_2/rnn/Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
d
"Predicates/CRF_2/rnn/range_1/startConst*
_output_shapes
: *
value	B :*
dtype0
d
"Predicates/CRF_2/rnn/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
?
Predicates/CRF_2/rnn/range_1Range"Predicates/CRF_2/rnn/range_1/startPredicates/CRF_2/rnn/Rank_1"Predicates/CRF_2/rnn/range_1/delta*
_output_shapes
:
w
&Predicates/CRF_2/rnn/concat_2/values_0Const*
dtype0*
valueB"       *
_output_shapes
:
d
"Predicates/CRF_2/rnn/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
Predicates/CRF_2/rnn/concat_2ConcatV2&Predicates/CRF_2/rnn/concat_2/values_0Predicates/CRF_2/rnn/range_1"Predicates/CRF_2/rnn/concat_2/axis*
T0*
_output_shapes
:*
N
?
 Predicates/CRF_2/rnn/transpose_1	Transpose9Predicates/CRF_2/rnn/TensorArrayStack/TensorArrayGatherV3Predicates/CRF_2/rnn/concat_2*
T0*,
_output_shapes
:??????????
?
 Predicates/CRF_2/ReverseSequenceReverseSequence Predicates/CRF_2/rnn/transpose_1Predicates/CRF_2/Maximum*
T0*
seq_dim*

Tlen0*,
_output_shapes
:??????????
c
!Predicates/CRF_2/ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
?
Predicates/CRF_2/ArgMaxArgMax!Predicates/CRF_2/rnn/while/Exit_3!Predicates/CRF_2/ArgMax/dimension*
T0*#
_output_shapes
:?????????
s
Predicates/CRF_2/CastCastPredicates/CRF_2/ArgMax*

SrcT0	*#
_output_shapes
:?????????*

DstT0
l
!Predicates/CRF_2/ExpandDims_1/dimConst*
dtype0*
valueB :
?????????*
_output_shapes
: 
?
Predicates/CRF_2/ExpandDims_1
ExpandDimsPredicates/CRF_2/Cast!Predicates/CRF_2/ExpandDims_1/dim*
T0*'
_output_shapes
:?????????
]
Predicates/CRF_2/rnn_1/RankConst*
_output_shapes
: *
value	B :*
dtype0
d
"Predicates/CRF_2/rnn_1/range/startConst*
_output_shapes
: *
value	B :*
dtype0
d
"Predicates/CRF_2/rnn_1/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
?
Predicates/CRF_2/rnn_1/rangeRange"Predicates/CRF_2/rnn_1/range/startPredicates/CRF_2/rnn_1/Rank"Predicates/CRF_2/rnn_1/range/delta*
_output_shapes
:
w
&Predicates/CRF_2/rnn_1/concat/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
d
"Predicates/CRF_2/rnn_1/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
Predicates/CRF_2/rnn_1/concatConcatV2&Predicates/CRF_2/rnn_1/concat/values_0Predicates/CRF_2/rnn_1/range"Predicates/CRF_2/rnn_1/concat/axis*
T0*
_output_shapes
:*
N
?
 Predicates/CRF_2/rnn_1/transpose	Transpose Predicates/CRF_2/ReverseSequencePredicates/CRF_2/rnn_1/concat*,
_output_shapes
:??????????*
T0
z
&Predicates/CRF_2/rnn_1/sequence_lengthIdentityPredicates/CRF_2/Maximum*
T0*#
_output_shapes
:?????????
l
Predicates/CRF_2/rnn_1/ShapeShape Predicates/CRF_2/rnn_1/transpose*
T0*
_output_shapes
:
t
*Predicates/CRF_2/rnn_1/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
v
,Predicates/CRF_2/rnn_1/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
v
,Predicates/CRF_2/rnn_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
$Predicates/CRF_2/rnn_1/strided_sliceStridedSlicePredicates/CRF_2/rnn_1/Shape*Predicates/CRF_2/rnn_1/strided_slice/stack,Predicates/CRF_2/rnn_1/strided_slice/stack_1,Predicates/CRF_2/rnn_1/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
t
Predicates/CRF_2/rnn_1/Shape_1Shape&Predicates/CRF_2/rnn_1/sequence_length*
T0*
_output_shapes
:
x
Predicates/CRF_2/rnn_1/stackPack$Predicates/CRF_2/rnn_1/strided_slice*
N*
_output_shapes
:*
T0
?
Predicates/CRF_2/rnn_1/EqualEqualPredicates/CRF_2/rnn_1/Shape_1Predicates/CRF_2/rnn_1/stack*
_output_shapes
:*
T0
f
Predicates/CRF_2/rnn_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
u
Predicates/CRF_2/rnn_1/AllAllPredicates/CRF_2/rnn_1/EqualPredicates/CRF_2/rnn_1/Const*
_output_shapes
: 
?
#Predicates/CRF_2/rnn_1/Assert/ConstConst*W
valueNBL BFExpected shape for Tensor Predicates/CRF_2/rnn_1/sequence_length:0 is *
_output_shapes
: *
dtype0
v
%Predicates/CRF_2/rnn_1/Assert/Const_1Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0
?
+Predicates/CRF_2/rnn_1/Assert/Assert/data_0Const*W
valueNBL BFExpected shape for Tensor Predicates/CRF_2/rnn_1/sequence_length:0 is *
_output_shapes
: *
dtype0
|
+Predicates/CRF_2/rnn_1/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
?
$Predicates/CRF_2/rnn_1/Assert/AssertAssertPredicates/CRF_2/rnn_1/All+Predicates/CRF_2/rnn_1/Assert/Assert/data_0Predicates/CRF_2/rnn_1/stack+Predicates/CRF_2/rnn_1/Assert/Assert/data_2Predicates/CRF_2/rnn_1/Shape_1*
T
2
?
"Predicates/CRF_2/rnn_1/CheckSeqLenIdentity&Predicates/CRF_2/rnn_1/sequence_length%^Predicates/CRF_2/rnn_1/Assert/Assert*
T0*#
_output_shapes
:?????????
n
Predicates/CRF_2/rnn_1/Shape_2Shape Predicates/CRF_2/rnn_1/transpose*
T0*
_output_shapes
:
v
,Predicates/CRF_2/rnn_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.Predicates/CRF_2/rnn_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.Predicates/CRF_2/rnn_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
&Predicates/CRF_2/rnn_1/strided_slice_1StridedSlicePredicates/CRF_2/rnn_1/Shape_2,Predicates/CRF_2/rnn_1/strided_slice_1/stack.Predicates/CRF_2/rnn_1/strided_slice_1/stack_1.Predicates/CRF_2/rnn_1/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
n
Predicates/CRF_2/rnn_1/Shape_3Shape Predicates/CRF_2/rnn_1/transpose*
T0*
_output_shapes
:
v
,Predicates/CRF_2/rnn_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
x
.Predicates/CRF_2/rnn_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.Predicates/CRF_2/rnn_1/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
&Predicates/CRF_2/rnn_1/strided_slice_2StridedSlicePredicates/CRF_2/rnn_1/Shape_3,Predicates/CRF_2/rnn_1/strided_slice_2/stack.Predicates/CRF_2/rnn_1/strided_slice_2/stack_1.Predicates/CRF_2/rnn_1/strided_slice_2/stack_2*
T0*
shrink_axis_mask*
_output_shapes
: *
Index0
g
%Predicates/CRF_2/rnn_1/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
?
!Predicates/CRF_2/rnn_1/ExpandDims
ExpandDims&Predicates/CRF_2/rnn_1/strided_slice_2%Predicates/CRF_2/rnn_1/ExpandDims/dim*
T0*
_output_shapes
:
h
Predicates/CRF_2/rnn_1/Const_1Const*
valueB:*
_output_shapes
:*
dtype0
f
$Predicates/CRF_2/rnn_1/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn_1/concat_1ConcatV2!Predicates/CRF_2/rnn_1/ExpandDimsPredicates/CRF_2/rnn_1/Const_1$Predicates/CRF_2/rnn_1/concat_1/axis*
T0*
N*
_output_shapes
:
d
"Predicates/CRF_2/rnn_1/zeros/ConstConst*
value	B : *
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn_1/zerosFillPredicates/CRF_2/rnn_1/concat_1"Predicates/CRF_2/rnn_1/zeros/Const*
T0*'
_output_shapes
:?????????
h
Predicates/CRF_2/rnn_1/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
?
Predicates/CRF_2/rnn_1/MinMin"Predicates/CRF_2/rnn_1/CheckSeqLenPredicates/CRF_2/rnn_1/Const_2*
T0*
_output_shapes
: 
h
Predicates/CRF_2/rnn_1/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
?
Predicates/CRF_2/rnn_1/MaxMax"Predicates/CRF_2/rnn_1/CheckSeqLenPredicates/CRF_2/rnn_1/Const_3*
T0*
_output_shapes
: 
]
Predicates/CRF_2/rnn_1/timeConst*
value	B : *
dtype0*
_output_shapes
: 
?
"Predicates/CRF_2/rnn_1/TensorArrayTensorArrayV3&Predicates/CRF_2/rnn_1/strided_slice_1*
_output_shapes

:: *B
tensor_array_name-+Predicates/CRF_2/rnn_1/dynamic_rnn/output_0*
dtype0*$
element_shape:?????????*
identical_element_shapes(
?
$Predicates/CRF_2/rnn_1/TensorArray_1TensorArrayV3&Predicates/CRF_2/rnn_1/strided_slice_1*$
element_shape:?????????*
identical_element_shapes(*A
tensor_array_name,*Predicates/CRF_2/rnn_1/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 

/Predicates/CRF_2/rnn_1/TensorArrayUnstack/ShapeShape Predicates/CRF_2/rnn_1/transpose*
T0*
_output_shapes
:
?
=Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
?Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
?Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
7Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_sliceStridedSlice/Predicates/CRF_2/rnn_1/TensorArrayUnstack/Shape=Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_slice/stack?Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_slice/stack_1?Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
w
5Predicates/CRF_2/rnn_1/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
w
5Predicates/CRF_2/rnn_1/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
/Predicates/CRF_2/rnn_1/TensorArrayUnstack/rangeRange5Predicates/CRF_2/rnn_1/TensorArrayUnstack/range/start7Predicates/CRF_2/rnn_1/TensorArrayUnstack/strided_slice5Predicates/CRF_2/rnn_1/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????
?
QPredicates/CRF_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3$Predicates/CRF_2/rnn_1/TensorArray_1/Predicates/CRF_2/rnn_1/TensorArrayUnstack/range Predicates/CRF_2/rnn_1/transpose&Predicates/CRF_2/rnn_1/TensorArray_1:1*
_output_shapes
: *3
_class)
'%loc:@Predicates/CRF_2/rnn_1/transpose*
T0
b
 Predicates/CRF_2/rnn_1/Maximum/xConst*
value	B :*
_output_shapes
: *
dtype0
?
Predicates/CRF_2/rnn_1/MaximumMaximum Predicates/CRF_2/rnn_1/Maximum/xPredicates/CRF_2/rnn_1/Max*
T0*
_output_shapes
: 
?
Predicates/CRF_2/rnn_1/MinimumMinimum&Predicates/CRF_2/rnn_1/strided_slice_1Predicates/CRF_2/rnn_1/Maximum*
_output_shapes
: *
T0
p
.Predicates/CRF_2/rnn_1/while/iteration_counterConst*
value	B : *
_output_shapes
: *
dtype0
?
"Predicates/CRF_2/rnn_1/while/EnterEnter.Predicates/CRF_2/rnn_1/while/iteration_counter*:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
?
$Predicates/CRF_2/rnn_1/while/Enter_1EnterPredicates/CRF_2/rnn_1/time*:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
?
$Predicates/CRF_2/rnn_1/while/Enter_2Enter$Predicates/CRF_2/rnn_1/TensorArray:1*
parallel_iterations *:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
T0*
_output_shapes
: 
?
$Predicates/CRF_2/rnn_1/while/Enter_3EnterPredicates/CRF_2/ExpandDims_1*
parallel_iterations *:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
T0*'
_output_shapes
:?????????
?
"Predicates/CRF_2/rnn_1/while/MergeMerge"Predicates/CRF_2/rnn_1/while/Enter*Predicates/CRF_2/rnn_1/while/NextIteration*
T0*
_output_shapes
: : *
N
?
$Predicates/CRF_2/rnn_1/while/Merge_1Merge$Predicates/CRF_2/rnn_1/while/Enter_1,Predicates/CRF_2/rnn_1/while/NextIteration_1*
T0*
_output_shapes
: : *
N
?
$Predicates/CRF_2/rnn_1/while/Merge_2Merge$Predicates/CRF_2/rnn_1/while/Enter_2,Predicates/CRF_2/rnn_1/while/NextIteration_2*
T0*
_output_shapes
: : *
N
?
$Predicates/CRF_2/rnn_1/while/Merge_3Merge$Predicates/CRF_2/rnn_1/while/Enter_3,Predicates/CRF_2/rnn_1/while/NextIteration_3*
T0*)
_output_shapes
:?????????: *
N
?
!Predicates/CRF_2/rnn_1/while/LessLess"Predicates/CRF_2/rnn_1/while/Merge'Predicates/CRF_2/rnn_1/while/Less/Enter*
T0*
_output_shapes
: 
?
'Predicates/CRF_2/rnn_1/while/Less/EnterEnter&Predicates/CRF_2/rnn_1/strided_slice_1*
is_constant(*
parallel_iterations *
T0*:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
_output_shapes
: 
?
#Predicates/CRF_2/rnn_1/while/Less_1Less$Predicates/CRF_2/rnn_1/while/Merge_1)Predicates/CRF_2/rnn_1/while/Less_1/Enter*
T0*
_output_shapes
: 
?
)Predicates/CRF_2/rnn_1/while/Less_1/EnterEnterPredicates/CRF_2/rnn_1/Minimum*
parallel_iterations *
_output_shapes
: *:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
is_constant(*
T0
?
'Predicates/CRF_2/rnn_1/while/LogicalAnd
LogicalAnd!Predicates/CRF_2/rnn_1/while/Less#Predicates/CRF_2/rnn_1/while/Less_1*
_output_shapes
: 
r
%Predicates/CRF_2/rnn_1/while/LoopCondLoopCond'Predicates/CRF_2/rnn_1/while/LogicalAnd*
_output_shapes
: 
?
#Predicates/CRF_2/rnn_1/while/SwitchSwitch"Predicates/CRF_2/rnn_1/while/Merge%Predicates/CRF_2/rnn_1/while/LoopCond*5
_class+
)'loc:@Predicates/CRF_2/rnn_1/while/Merge*
_output_shapes
: : *
T0
?
%Predicates/CRF_2/rnn_1/while/Switch_1Switch$Predicates/CRF_2/rnn_1/while/Merge_1%Predicates/CRF_2/rnn_1/while/LoopCond*7
_class-
+)loc:@Predicates/CRF_2/rnn_1/while/Merge_1*
_output_shapes
: : *
T0
?
%Predicates/CRF_2/rnn_1/while/Switch_2Switch$Predicates/CRF_2/rnn_1/while/Merge_2%Predicates/CRF_2/rnn_1/while/LoopCond*7
_class-
+)loc:@Predicates/CRF_2/rnn_1/while/Merge_2*
_output_shapes
: : *
T0
?
%Predicates/CRF_2/rnn_1/while/Switch_3Switch$Predicates/CRF_2/rnn_1/while/Merge_3%Predicates/CRF_2/rnn_1/while/LoopCond*7
_class-
+)loc:@Predicates/CRF_2/rnn_1/while/Merge_3*:
_output_shapes(
&:?????????:?????????*
T0
y
%Predicates/CRF_2/rnn_1/while/IdentityIdentity%Predicates/CRF_2/rnn_1/while/Switch:1*
_output_shapes
: *
T0
}
'Predicates/CRF_2/rnn_1/while/Identity_1Identity'Predicates/CRF_2/rnn_1/while/Switch_1:1*
T0*
_output_shapes
: 
}
'Predicates/CRF_2/rnn_1/while/Identity_2Identity'Predicates/CRF_2/rnn_1/while/Switch_2:1*
T0*
_output_shapes
: 
?
'Predicates/CRF_2/rnn_1/while/Identity_3Identity'Predicates/CRF_2/rnn_1/while/Switch_3:1*
T0*'
_output_shapes
:?????????
?
"Predicates/CRF_2/rnn_1/while/add/yConst&^Predicates/CRF_2/rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
?
 Predicates/CRF_2/rnn_1/while/addAdd%Predicates/CRF_2/rnn_1/while/Identity"Predicates/CRF_2/rnn_1/while/add/y*
T0*
_output_shapes
: 
?
.Predicates/CRF_2/rnn_1/while/TensorArrayReadV3TensorArrayReadV34Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/Enter'Predicates/CRF_2/rnn_1/while/Identity_16Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:?????????
?
4Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/EnterEnter$Predicates/CRF_2/rnn_1/TensorArray_1*
parallel_iterations *
T0*
_output_shapes
:*:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
is_constant(
?
6Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/Enter_1EnterQPredicates/CRF_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
T0*
_output_shapes
: *:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
is_constant(
?
)Predicates/CRF_2/rnn_1/while/GreaterEqualGreaterEqual'Predicates/CRF_2/rnn_1/while/Identity_1/Predicates/CRF_2/rnn_1/while/GreaterEqual/Enter*#
_output_shapes
:?????????*
T0
?
/Predicates/CRF_2/rnn_1/while/GreaterEqual/EnterEnter"Predicates/CRF_2/rnn_1/CheckSeqLen*
parallel_iterations *
T0*#
_output_shapes
:?????????*:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
is_constant(
?
$Predicates/CRF_2/rnn_1/while/SqueezeSqueeze'Predicates/CRF_2/rnn_1/while/Identity_3*#
_output_shapes
:?????????*
T0*
squeeze_dims

?
"Predicates/CRF_2/rnn_1/while/ShapeShape.Predicates/CRF_2/rnn_1/while/TensorArrayReadV3*
_output_shapes
:*
T0
?
0Predicates/CRF_2/rnn_1/while/strided_slice/stackConst&^Predicates/CRF_2/rnn_1/while/Identity*
_output_shapes
:*
dtype0*
valueB: 
?
2Predicates/CRF_2/rnn_1/while/strided_slice/stack_1Const&^Predicates/CRF_2/rnn_1/while/Identity*
_output_shapes
:*
dtype0*
valueB:
?
2Predicates/CRF_2/rnn_1/while/strided_slice/stack_2Const&^Predicates/CRF_2/rnn_1/while/Identity*
_output_shapes
:*
dtype0*
valueB:
?
*Predicates/CRF_2/rnn_1/while/strided_sliceStridedSlice"Predicates/CRF_2/rnn_1/while/Shape0Predicates/CRF_2/rnn_1/while/strided_slice/stack2Predicates/CRF_2/rnn_1/while/strided_slice/stack_12Predicates/CRF_2/rnn_1/while/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
?
(Predicates/CRF_2/rnn_1/while/range/startConst&^Predicates/CRF_2/rnn_1/while/Identity*
_output_shapes
: *
dtype0*
value	B : 
?
(Predicates/CRF_2/rnn_1/while/range/deltaConst&^Predicates/CRF_2/rnn_1/while/Identity*
dtype0*
_output_shapes
: *
value	B :
?
"Predicates/CRF_2/rnn_1/while/rangeRange(Predicates/CRF_2/rnn_1/while/range/start*Predicates/CRF_2/rnn_1/while/strided_slice(Predicates/CRF_2/rnn_1/while/range/delta*#
_output_shapes
:?????????
?
"Predicates/CRF_2/rnn_1/while/stackPack"Predicates/CRF_2/rnn_1/while/range$Predicates/CRF_2/rnn_1/while/Squeeze*

axis*
T0*'
_output_shapes
:?????????*
N
?
%Predicates/CRF_2/rnn_1/while/GatherNdGatherNd.Predicates/CRF_2/rnn_1/while/TensorArrayReadV3"Predicates/CRF_2/rnn_1/while/stack*#
_output_shapes
:?????????*
Tparams0*
Tindices0
?
+Predicates/CRF_2/rnn_1/while/ExpandDims/dimConst&^Predicates/CRF_2/rnn_1/while/Identity*
dtype0*
_output_shapes
: *
valueB :
?????????
?
'Predicates/CRF_2/rnn_1/while/ExpandDims
ExpandDims%Predicates/CRF_2/rnn_1/while/GatherNd+Predicates/CRF_2/rnn_1/while/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
#Predicates/CRF_2/rnn_1/while/SelectSelect)Predicates/CRF_2/rnn_1/while/GreaterEqual)Predicates/CRF_2/rnn_1/while/Select/Enter'Predicates/CRF_2/rnn_1/while/ExpandDims*
T0*'
_output_shapes
:?????????*:
_class0
.,loc:@Predicates/CRF_2/rnn_1/while/ExpandDims
?
)Predicates/CRF_2/rnn_1/while/Select/EnterEnterPredicates/CRF_2/rnn_1/zeros*'
_output_shapes
:?????????*:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*
T0*
is_constant(*
parallel_iterations *:
_class0
.,loc:@Predicates/CRF_2/rnn_1/while/ExpandDims
?
%Predicates/CRF_2/rnn_1/while/Select_1Select)Predicates/CRF_2/rnn_1/while/GreaterEqual'Predicates/CRF_2/rnn_1/while/Identity_3'Predicates/CRF_2/rnn_1/while/ExpandDims*
T0*'
_output_shapes
:?????????*:
_class0
.,loc:@Predicates/CRF_2/rnn_1/while/ExpandDims
?
@Predicates/CRF_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3FPredicates/CRF_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter'Predicates/CRF_2/rnn_1/while/Identity_1#Predicates/CRF_2/rnn_1/while/Select'Predicates/CRF_2/rnn_1/while/Identity_2*
_output_shapes
: *:
_class0
.,loc:@Predicates/CRF_2/rnn_1/while/ExpandDims*
T0
?
FPredicates/CRF_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter"Predicates/CRF_2/rnn_1/TensorArray*
_output_shapes
:*:

frame_name,*Predicates/CRF_2/rnn_1/while/while_context*:
_class0
.,loc:@Predicates/CRF_2/rnn_1/while/ExpandDims*
parallel_iterations *
T0*
is_constant(
?
$Predicates/CRF_2/rnn_1/while/add_1/yConst&^Predicates/CRF_2/rnn_1/while/Identity*
_output_shapes
: *
dtype0*
value	B :
?
"Predicates/CRF_2/rnn_1/while/add_1Add'Predicates/CRF_2/rnn_1/while/Identity_1$Predicates/CRF_2/rnn_1/while/add_1/y*
_output_shapes
: *
T0
~
*Predicates/CRF_2/rnn_1/while/NextIterationNextIteration Predicates/CRF_2/rnn_1/while/add*
_output_shapes
: *
T0
?
,Predicates/CRF_2/rnn_1/while/NextIteration_1NextIteration"Predicates/CRF_2/rnn_1/while/add_1*
_output_shapes
: *
T0
?
,Predicates/CRF_2/rnn_1/while/NextIteration_2NextIteration@Predicates/CRF_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
?
,Predicates/CRF_2/rnn_1/while/NextIteration_3NextIteration%Predicates/CRF_2/rnn_1/while/Select_1*'
_output_shapes
:?????????*
T0
o
!Predicates/CRF_2/rnn_1/while/ExitExit#Predicates/CRF_2/rnn_1/while/Switch*
_output_shapes
: *
T0
s
#Predicates/CRF_2/rnn_1/while/Exit_1Exit%Predicates/CRF_2/rnn_1/while/Switch_1*
_output_shapes
: *
T0
s
#Predicates/CRF_2/rnn_1/while/Exit_2Exit%Predicates/CRF_2/rnn_1/while/Switch_2*
T0*
_output_shapes
: 
?
#Predicates/CRF_2/rnn_1/while/Exit_3Exit%Predicates/CRF_2/rnn_1/while/Switch_3*
T0*'
_output_shapes
:?????????
?
9Predicates/CRF_2/rnn_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3"Predicates/CRF_2/rnn_1/TensorArray#Predicates/CRF_2/rnn_1/while/Exit_2*
_output_shapes
: *5
_class+
)'loc:@Predicates/CRF_2/rnn_1/TensorArray
?
3Predicates/CRF_2/rnn_1/TensorArrayStack/range/startConst*
value	B : *
dtype0*5
_class+
)'loc:@Predicates/CRF_2/rnn_1/TensorArray*
_output_shapes
: 
?
3Predicates/CRF_2/rnn_1/TensorArrayStack/range/deltaConst*
value	B :*
dtype0*5
_class+
)'loc:@Predicates/CRF_2/rnn_1/TensorArray*
_output_shapes
: 
?
-Predicates/CRF_2/rnn_1/TensorArrayStack/rangeRange3Predicates/CRF_2/rnn_1/TensorArrayStack/range/start9Predicates/CRF_2/rnn_1/TensorArrayStack/TensorArraySizeV33Predicates/CRF_2/rnn_1/TensorArrayStack/range/delta*5
_class+
)'loc:@Predicates/CRF_2/rnn_1/TensorArray*#
_output_shapes
:?????????
?
;Predicates/CRF_2/rnn_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3"Predicates/CRF_2/rnn_1/TensorArray-Predicates/CRF_2/rnn_1/TensorArrayStack/range#Predicates/CRF_2/rnn_1/while/Exit_2*
dtype0*$
element_shape:?????????*,
_output_shapes
:??????????*5
_class+
)'loc:@Predicates/CRF_2/rnn_1/TensorArray
h
Predicates/CRF_2/rnn_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
_
Predicates/CRF_2/rnn_1/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
f
$Predicates/CRF_2/rnn_1/range_1/startConst*
dtype0*
value	B :*
_output_shapes
: 
f
$Predicates/CRF_2/rnn_1/range_1/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
?
Predicates/CRF_2/rnn_1/range_1Range$Predicates/CRF_2/rnn_1/range_1/startPredicates/CRF_2/rnn_1/Rank_1$Predicates/CRF_2/rnn_1/range_1/delta*
_output_shapes
:
y
(Predicates/CRF_2/rnn_1/concat_2/values_0Const*
dtype0*
valueB"       *
_output_shapes
:
f
$Predicates/CRF_2/rnn_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
Predicates/CRF_2/rnn_1/concat_2ConcatV2(Predicates/CRF_2/rnn_1/concat_2/values_0Predicates/CRF_2/rnn_1/range_1$Predicates/CRF_2/rnn_1/concat_2/axis*
T0*
N*
_output_shapes
:
?
"Predicates/CRF_2/rnn_1/transpose_1	Transpose;Predicates/CRF_2/rnn_1/TensorArrayStack/TensorArrayGatherV3Predicates/CRF_2/rnn_1/concat_2*
T0*,
_output_shapes
:??????????
?
Predicates/CRF_2/Squeeze_1Squeeze"Predicates/CRF_2/rnn_1/transpose_1*
T0*
squeeze_dims
*(
_output_shapes
:??????????
^
Predicates/CRF_2/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
?
Predicates/CRF_2/concatConcatV2Predicates/CRF_2/ExpandDims_1Predicates/CRF_2/Squeeze_1Predicates/CRF_2/concat/axis*
T0*
N*(
_output_shapes
:??????????
?
"Predicates/CRF_2/ReverseSequence_1ReverseSequencePredicates/CRF_2/concatpred_original_sequence_lengths*
T0*
seq_dim*

Tlen0*(
_output_shapes
:??????????
h
&Predicates/CRF_2/Max/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
?
Predicates/CRF_2/MaxMax!Predicates/CRF_2/rnn/while/Exit_3&Predicates/CRF_2/Max/reduction_indices*#
_output_shapes
:?????????*
T0
~
Predicates/ToInt64Cast Predicates/CRF/ReverseSequence_1*

DstT0	*(
_output_shapes
:??????????*

SrcT0
?
.Predicates/hash_table_Lookup/LookupTableFindV2LookupTableFindV2%Predicates/index_to_string/hash_tablePredicates/ToInt64 Predicates/index_to_string/Const*(
_output_shapes
:??????????*

Tout0*	
Tin0	
b
Predicates/viterbiIdentityPredicates/CRF_2/Max*#
_output_shapes
:?????????*
T0

initNoOp
]
init_all_tablesNoOpD^Predicates/index_to_string/table_init/InitializeTableFromTextFileV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
?
save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_564d359d2de9493baa5e94f439bd076a/part*
dtype0
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?B#Predicates/BiLSTM/bw/lstm_cell/biasB%Predicates/BiLSTM/bw/lstm_cell/kernelB#Predicates/BiLSTM/fw/lstm_cell/biasB%Predicates/BiLSTM/fw/lstm_cell/kernelBPredicates/CRF/transitionBPredicates/CRF_2/transition_2BPredicates/dep/VariableBPredicates/mask/VariableBPredicates/pos/VariableBPredicates/pred_arc/VariableBPredicates/proj/WBPredicates/proj/bBPredicates/proj_2/W_2BPredicates/proj_2/b_2Bglobal_step*
_output_shapes
:*
dtype0
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices#Predicates/BiLSTM/bw/lstm_cell/bias%Predicates/BiLSTM/bw/lstm_cell/kernel#Predicates/BiLSTM/fw/lstm_cell/bias%Predicates/BiLSTM/fw/lstm_cell/kernelPredicates/CRF/transitionPredicates/CRF_2/transition_2Predicates/dep/VariablePredicates/mask/VariablePredicates/pos/VariablePredicates/pred_arc/VariablePredicates/proj/WPredicates/proj/bPredicates/proj_2/W_2Predicates/proj_2/b_2global_step"/device:CPU:0*
dtypes
2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*?
value?B?B#Predicates/BiLSTM/bw/lstm_cell/biasB%Predicates/BiLSTM/bw/lstm_cell/kernelB#Predicates/BiLSTM/fw/lstm_cell/biasB%Predicates/BiLSTM/fw/lstm_cell/kernelBPredicates/CRF/transitionBPredicates/CRF_2/transition_2BPredicates/dep/VariableBPredicates/mask/VariableBPredicates/pos/VariableBPredicates/pred_arc/VariableBPredicates/proj/WBPredicates/proj/bBPredicates/proj_2/W_2BPredicates/proj_2/b_2Bglobal_step*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*1
value(B&B B B B B B B B B B B B B B B *
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	
?
save/AssignAssign#Predicates/BiLSTM/bw/lstm_cell/biassave/RestoreV2*
_output_shapes	
:?*6
_class,
*(loc:@Predicates/BiLSTM/bw/lstm_cell/bias*
T0
?
save/Assign_1Assign%Predicates/BiLSTM/bw/lstm_cell/kernelsave/RestoreV2:1* 
_output_shapes
:
??*8
_class.
,*loc:@Predicates/BiLSTM/bw/lstm_cell/kernel*
T0
?
save/Assign_2Assign#Predicates/BiLSTM/fw/lstm_cell/biassave/RestoreV2:2*6
_class,
*(loc:@Predicates/BiLSTM/fw/lstm_cell/bias*
_output_shapes	
:?*
T0
?
save/Assign_3Assign%Predicates/BiLSTM/fw/lstm_cell/kernelsave/RestoreV2:3*8
_class.
,*loc:@Predicates/BiLSTM/fw/lstm_cell/kernel* 
_output_shapes
:
??*
T0
?
save/Assign_4AssignPredicates/CRF/transitionsave/RestoreV2:4*,
_class"
 loc:@Predicates/CRF/transition*
T0*
_output_shapes

:
?
save/Assign_5AssignPredicates/CRF_2/transition_2save/RestoreV2:5*
_output_shapes

:*0
_class&
$"loc:@Predicates/CRF_2/transition_2*
T0
?
save/Assign_6AssignPredicates/dep/Variablesave/RestoreV2:6**
_class 
loc:@Predicates/dep/Variable*
_output_shapes

:>*
T0
?
save/Assign_7AssignPredicates/mask/Variablesave/RestoreV2:7*
_output_shapes

:*+
_class!
loc:@Predicates/mask/Variable*
T0
?
save/Assign_8AssignPredicates/pos/Variablesave/RestoreV2:8**
_class 
loc:@Predicates/pos/Variable*
_output_shapes

:*
T0
?
save/Assign_9AssignPredicates/pred_arc/Variablesave/RestoreV2:9*
_output_shapes

:*/
_class%
#!loc:@Predicates/pred_arc/Variable*
T0
?
save/Assign_10AssignPredicates/proj/Wsave/RestoreV2:10*
_output_shapes
:	?*$
_class
loc:@Predicates/proj/W*
T0
?
save/Assign_11AssignPredicates/proj/bsave/RestoreV2:11*
T0*$
_class
loc:@Predicates/proj/b*
_output_shapes
:
?
save/Assign_12AssignPredicates/proj_2/W_2save/RestoreV2:12*
T0*(
_class
loc:@Predicates/proj_2/W_2*
_output_shapes
:	?
?
save/Assign_13AssignPredicates/proj_2/b_2save/RestoreV2:13*
T0*(
_class
loc:@Predicates/proj_2/b_2*
_output_shapes
:
y
save/Assign_14Assignglobal_stepsave/RestoreV2:14*
T0	*
_class
loc:@global_step*
_output_shapes
: 
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"?
	variables??
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
~
Predicates/pos/Variable:0Predicates/pos/Variable/AssignPredicates/pos/Variable/read:02Predicates/pos/random_uniform:08
~
Predicates/dep/Variable:0Predicates/dep/Variable/AssignPredicates/dep/Variable/read:02Predicates/dep/random_uniform:08
?
Predicates/mask/Variable:0Predicates/mask/Variable/AssignPredicates/mask/Variable/read:02 Predicates/mask/random_uniform:08
?
'Predicates/BiLSTM/fw/lstm_cell/kernel:0,Predicates/BiLSTM/fw/lstm_cell/kernel/Assign,Predicates/BiLSTM/fw/lstm_cell/kernel/read:02BPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform:08
?
%Predicates/BiLSTM/fw/lstm_cell/bias:0*Predicates/BiLSTM/fw/lstm_cell/bias/Assign*Predicates/BiLSTM/fw/lstm_cell/bias/read:027Predicates/BiLSTM/fw/lstm_cell/bias/Initializer/Const:08
?
'Predicates/BiLSTM/bw/lstm_cell/kernel:0,Predicates/BiLSTM/bw/lstm_cell/kernel/Assign,Predicates/BiLSTM/bw/lstm_cell/kernel/read:02BPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform:08
?
%Predicates/BiLSTM/bw/lstm_cell/bias:0*Predicates/BiLSTM/bw/lstm_cell/bias/Assign*Predicates/BiLSTM/bw/lstm_cell/bias/read:027Predicates/BiLSTM/bw/lstm_cell/bias/Initializer/Const:08
r
Predicates/proj/W:0Predicates/proj/W/AssignPredicates/proj/W/read:02%Predicates/proj/W/Initializer/mul_1:08
r
Predicates/proj/b:0Predicates/proj/b/AssignPredicates/proj/b/read:02%Predicates/proj/b/Initializer/zeros:08
?
Predicates/CRF/transition:0 Predicates/CRF/transition/Assign Predicates/CRF/transition/read:026Predicates/CRF/transition/Initializer/random_uniform:08
?
Predicates/pred_arc/Variable:0#Predicates/pred_arc/Variable/Assign#Predicates/pred_arc/Variable/read:02$Predicates/pred_arc/random_uniform:08
?
Predicates/proj_2/W_2:0Predicates/proj_2/W_2/AssignPredicates/proj_2/W_2/read:02)Predicates/proj_2/W_2/Initializer/mul_1:08
?
Predicates/proj_2/b_2:0Predicates/proj_2/b_2/AssignPredicates/proj_2/b_2/read:02)Predicates/proj_2/b_2/Initializer/zeros:08
?
Predicates/CRF_2/transition_2:0$Predicates/CRF_2/transition_2/Assign$Predicates/CRF_2/transition_2/read:02:Predicates/CRF_2/transition_2/Initializer/random_uniform:08"??
while_context????
?.
2Predicates/BiLSTM/BiLSTM/fw/fw/while/while_context */Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCond:02,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge:0:/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity:0B+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit:0B-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_1:0B-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_2:0B-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_3:0B-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_4:0J?(
,Predicates/BiLSTM/BiLSTM/fw/fw/CheckSeqLen:0
(Predicates/BiLSTM/BiLSTM/fw/fw/Minimum:0
,Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray:0
[Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
.Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray_1:0
0Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1:0
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_1:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_2:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_3:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_4:0
+Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_1:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_2:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_3:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Exit_4:0
9Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Enter:0
3Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity:0
1Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_1:0
1Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_2:0
1Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_3:0
1Predicates/BiLSTM/BiLSTM/fw/fw/while/Identity_4:0
1Predicates/BiLSTM/BiLSTM/fw/fw/while/Less/Enter:0
+Predicates/BiLSTM/BiLSTM/fw/fw/while/Less:0
3Predicates/BiLSTM/BiLSTM/fw/fw/while/Less_1/Enter:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Less_1:0
1Predicates/BiLSTM/BiLSTM/fw/fw/while/LogicalAnd:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/LoopCond:0
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge:0
,Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge:1
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_1:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_1:1
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_2:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_2:1
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_3:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_3:1
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_4:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/Merge_4:1
4Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration:0
6Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_1:0
6Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_2:0
6Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_3:0
6Predicates/BiLSTM/BiLSTM/fw/fw/while/NextIteration_4:0
3Predicates/BiLSTM/BiLSTM/fw/fw/while/Select/Enter:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Select:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Select_1:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Select_2:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch:0
-Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch:1
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_1:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_1:1
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_2:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_2:1
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_3:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_3:1
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_4:0
/Predicates/BiLSTM/BiLSTM/fw/fw/while/Switch_4:1
>Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter:0
@Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1:0
8Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3:0
PPredicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
JPredicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
,Predicates/BiLSTM/BiLSTM/fw/fw/while/add/y:0
*Predicates/BiLSTM/BiLSTM/fw/fw/while/add:0
.Predicates/BiLSTM/BiLSTM/fw/fw/while/add_1/y:0
,Predicates/BiLSTM/BiLSTM/fw/fw/while/add_1:0
DPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter:0
FPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1:0
>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:0
>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:1
>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:2
>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:3
>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:4
>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:5
>Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell:6
6Predicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/zeros:0
&Predicates/BiLSTM/BiLSTM/fw/fw/zeros:0
*Predicates/BiLSTM/fw/lstm_cell/bias/read:0
,Predicates/BiLSTM/fw/lstm_cell/kernel/read:0_
(Predicates/BiLSTM/BiLSTM/fw/fw/Minimum:03Predicates/BiLSTM/BiLSTM/fw/fw/while/Less_1/Enter:0?
[Predicates/BiLSTM/BiLSTM/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0@Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter_1:0t
*Predicates/BiLSTM/fw/lstm_cell/bias/read:0FPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter_1:0?
,Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray:0PPredicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0]
&Predicates/BiLSTM/BiLSTM/fw/fw/zeros:03Predicates/BiLSTM/BiLSTM/fw/fw/while/Select/Enter:0t
,Predicates/BiLSTM/fw/lstm_cell/kernel/read:0DPredicates/BiLSTM/BiLSTM/fw/fw/while/lstm_cell/LSTMBlockCell/Enter:0p
.Predicates/BiLSTM/BiLSTM/fw/fw/TensorArray_1:0>Predicates/BiLSTM/BiLSTM/fw/fw/while/TensorArrayReadV3/Enter:0i
,Predicates/BiLSTM/BiLSTM/fw/fw/CheckSeqLen:09Predicates/BiLSTM/BiLSTM/fw/fw/while/GreaterEqual/Enter:0e
0Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1:01Predicates/BiLSTM/BiLSTM/fw/fw/while/Less/Enter:0R,Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter:0R.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_1:0R.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_2:0R.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_3:0R.Predicates/BiLSTM/BiLSTM/fw/fw/while/Enter_4:0Z0Predicates/BiLSTM/BiLSTM/fw/fw/strided_slice_1:0
?.
2Predicates/BiLSTM/BiLSTM/bw/bw/while/while_context */Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCond:02,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge:0:/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity:0B+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit:0B-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_1:0B-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_2:0B-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_3:0B-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_4:0J?(
,Predicates/BiLSTM/BiLSTM/bw/bw/CheckSeqLen:0
(Predicates/BiLSTM/BiLSTM/bw/bw/Minimum:0
,Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray:0
[Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
.Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray_1:0
0Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1:0
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_1:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_2:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_3:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_4:0
+Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_1:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_2:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_3:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Exit_4:0
9Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Enter:0
3Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity:0
1Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_1:0
1Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_2:0
1Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_3:0
1Predicates/BiLSTM/BiLSTM/bw/bw/while/Identity_4:0
1Predicates/BiLSTM/BiLSTM/bw/bw/while/Less/Enter:0
+Predicates/BiLSTM/BiLSTM/bw/bw/while/Less:0
3Predicates/BiLSTM/BiLSTM/bw/bw/while/Less_1/Enter:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Less_1:0
1Predicates/BiLSTM/BiLSTM/bw/bw/while/LogicalAnd:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/LoopCond:0
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge:0
,Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge:1
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_1:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_1:1
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_2:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_2:1
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_3:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_3:1
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_4:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/Merge_4:1
4Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration:0
6Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_1:0
6Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_2:0
6Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_3:0
6Predicates/BiLSTM/BiLSTM/bw/bw/while/NextIteration_4:0
3Predicates/BiLSTM/BiLSTM/bw/bw/while/Select/Enter:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Select:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Select_1:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Select_2:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch:0
-Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch:1
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_1:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_1:1
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_2:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_2:1
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_3:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_3:1
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_4:0
/Predicates/BiLSTM/BiLSTM/bw/bw/while/Switch_4:1
>Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter:0
@Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1:0
8Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3:0
PPredicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
JPredicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
,Predicates/BiLSTM/BiLSTM/bw/bw/while/add/y:0
*Predicates/BiLSTM/BiLSTM/bw/bw/while/add:0
.Predicates/BiLSTM/BiLSTM/bw/bw/while/add_1/y:0
,Predicates/BiLSTM/BiLSTM/bw/bw/while/add_1:0
DPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter:0
FPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1:0
>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:0
>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:1
>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:2
>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:3
>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:4
>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:5
>Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell:6
6Predicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/zeros:0
&Predicates/BiLSTM/BiLSTM/bw/bw/zeros:0
*Predicates/BiLSTM/bw/lstm_cell/bias/read:0
,Predicates/BiLSTM/bw/lstm_cell/kernel/read:0?
,Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray:0PPredicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0_
(Predicates/BiLSTM/BiLSTM/bw/bw/Minimum:03Predicates/BiLSTM/BiLSTM/bw/bw/while/Less_1/Enter:0t
*Predicates/BiLSTM/bw/lstm_cell/bias/read:0FPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter_1:0e
0Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1:01Predicates/BiLSTM/BiLSTM/bw/bw/while/Less/Enter:0?
[Predicates/BiLSTM/BiLSTM/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0@Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter_1:0p
.Predicates/BiLSTM/BiLSTM/bw/bw/TensorArray_1:0>Predicates/BiLSTM/BiLSTM/bw/bw/while/TensorArrayReadV3/Enter:0i
,Predicates/BiLSTM/BiLSTM/bw/bw/CheckSeqLen:09Predicates/BiLSTM/BiLSTM/bw/bw/while/GreaterEqual/Enter:0]
&Predicates/BiLSTM/BiLSTM/bw/bw/zeros:03Predicates/BiLSTM/BiLSTM/bw/bw/while/Select/Enter:0t
,Predicates/BiLSTM/bw/lstm_cell/kernel/read:0DPredicates/BiLSTM/BiLSTM/bw/bw/while/lstm_cell/LSTMBlockCell/Enter:0R,Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter:0R.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_1:0R.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_2:0R.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_3:0R.Predicates/BiLSTM/BiLSTM/bw/bw/while/Enter_4:0Z0Predicates/BiLSTM/BiLSTM/bw/bw/strided_slice_1:0
?
&Predicates/CRF/rnn/while/while_context *#Predicates/CRF/rnn/while/LoopCond:02 Predicates/CRF/rnn/while/Merge:0:#Predicates/CRF/rnn/while/Identity:0BPredicates/CRF/rnn/while/Exit:0B!Predicates/CRF/rnn/while/Exit_1:0B!Predicates/CRF/rnn/while/Exit_2:0B!Predicates/CRF/rnn/while/Exit_3:0J?
Predicates/CRF/ExpandDims:0
 Predicates/CRF/rnn/CheckSeqLen:0
Predicates/CRF/rnn/Minimum:0
 Predicates/CRF/rnn/TensorArray:0
OPredicates/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
"Predicates/CRF/rnn/TensorArray_1:0
$Predicates/CRF/rnn/strided_slice_1:0
+Predicates/CRF/rnn/while/ArgMax/dimension:0
!Predicates/CRF/rnn/while/ArgMax:0
Predicates/CRF/rnn/while/Cast:0
 Predicates/CRF/rnn/while/Enter:0
"Predicates/CRF/rnn/while/Enter_1:0
"Predicates/CRF/rnn/while/Enter_2:0
"Predicates/CRF/rnn/while/Enter_3:0
Predicates/CRF/rnn/while/Exit:0
!Predicates/CRF/rnn/while/Exit_1:0
!Predicates/CRF/rnn/while/Exit_2:0
!Predicates/CRF/rnn/while/Exit_3:0
)Predicates/CRF/rnn/while/ExpandDims/dim:0
%Predicates/CRF/rnn/while/ExpandDims:0
-Predicates/CRF/rnn/while/GreaterEqual/Enter:0
'Predicates/CRF/rnn/while/GreaterEqual:0
#Predicates/CRF/rnn/while/Identity:0
%Predicates/CRF/rnn/while/Identity_1:0
%Predicates/CRF/rnn/while/Identity_2:0
%Predicates/CRF/rnn/while/Identity_3:0
%Predicates/CRF/rnn/while/Less/Enter:0
Predicates/CRF/rnn/while/Less:0
'Predicates/CRF/rnn/while/Less_1/Enter:0
!Predicates/CRF/rnn/while/Less_1:0
%Predicates/CRF/rnn/while/LogicalAnd:0
#Predicates/CRF/rnn/while/LoopCond:0
0Predicates/CRF/rnn/while/Max/reduction_indices:0
Predicates/CRF/rnn/while/Max:0
 Predicates/CRF/rnn/while/Merge:0
 Predicates/CRF/rnn/while/Merge:1
"Predicates/CRF/rnn/while/Merge_1:0
"Predicates/CRF/rnn/while/Merge_1:1
"Predicates/CRF/rnn/while/Merge_2:0
"Predicates/CRF/rnn/while/Merge_2:1
"Predicates/CRF/rnn/while/Merge_3:0
"Predicates/CRF/rnn/while/Merge_3:1
(Predicates/CRF/rnn/while/NextIteration:0
*Predicates/CRF/rnn/while/NextIteration_1:0
*Predicates/CRF/rnn/while/NextIteration_2:0
*Predicates/CRF/rnn/while/NextIteration_3:0
'Predicates/CRF/rnn/while/Select/Enter:0
!Predicates/CRF/rnn/while/Select:0
#Predicates/CRF/rnn/while/Select_1:0
!Predicates/CRF/rnn/while/Switch:0
!Predicates/CRF/rnn/while/Switch:1
#Predicates/CRF/rnn/while/Switch_1:0
#Predicates/CRF/rnn/while/Switch_1:1
#Predicates/CRF/rnn/while/Switch_2:0
#Predicates/CRF/rnn/while/Switch_2:1
#Predicates/CRF/rnn/while/Switch_3:0
#Predicates/CRF/rnn/while/Switch_3:1
2Predicates/CRF/rnn/while/TensorArrayReadV3/Enter:0
4Predicates/CRF/rnn/while/TensorArrayReadV3/Enter_1:0
,Predicates/CRF/rnn/while/TensorArrayReadV3:0
DPredicates/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
>Predicates/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
 Predicates/CRF/rnn/while/add/y:0
Predicates/CRF/rnn/while/add:0
&Predicates/CRF/rnn/while/add_1/Enter:0
 Predicates/CRF/rnn/while/add_1:0
 Predicates/CRF/rnn/while/add_2:0
"Predicates/CRF/rnn/while/add_3/y:0
 Predicates/CRF/rnn/while/add_3:0
Predicates/CRF/rnn/zeros:0G
Predicates/CRF/rnn/Minimum:0'Predicates/CRF/rnn/while/Less_1/Enter:0?
OPredicates/CRF/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:04Predicates/CRF/rnn/while/TensorArrayReadV3/Enter_1:0X
"Predicates/CRF/rnn/TensorArray_1:02Predicates/CRF/rnn/while/TensorArrayReadV3/Enter:0h
 Predicates/CRF/rnn/TensorArray:0DPredicates/CRF/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0E
Predicates/CRF/ExpandDims:0&Predicates/CRF/rnn/while/add_1/Enter:0E
Predicates/CRF/rnn/zeros:0'Predicates/CRF/rnn/while/Select/Enter:0Q
 Predicates/CRF/rnn/CheckSeqLen:0-Predicates/CRF/rnn/while/GreaterEqual/Enter:0M
$Predicates/CRF/rnn/strided_slice_1:0%Predicates/CRF/rnn/while/Less/Enter:0R Predicates/CRF/rnn/while/Enter:0R"Predicates/CRF/rnn/while/Enter_1:0R"Predicates/CRF/rnn/while/Enter_2:0R"Predicates/CRF/rnn/while/Enter_3:0Z$Predicates/CRF/rnn/strided_slice_1:0
? 
(Predicates/CRF/rnn_1/while/while_context *%Predicates/CRF/rnn_1/while/LoopCond:02"Predicates/CRF/rnn_1/while/Merge:0:%Predicates/CRF/rnn_1/while/Identity:0B!Predicates/CRF/rnn_1/while/Exit:0B#Predicates/CRF/rnn_1/while/Exit_1:0B#Predicates/CRF/rnn_1/while/Exit_2:0B#Predicates/CRF/rnn_1/while/Exit_3:0J?
"Predicates/CRF/rnn_1/CheckSeqLen:0
Predicates/CRF/rnn_1/Minimum:0
"Predicates/CRF/rnn_1/TensorArray:0
QPredicates/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
$Predicates/CRF/rnn_1/TensorArray_1:0
&Predicates/CRF/rnn_1/strided_slice_1:0
"Predicates/CRF/rnn_1/while/Enter:0
$Predicates/CRF/rnn_1/while/Enter_1:0
$Predicates/CRF/rnn_1/while/Enter_2:0
$Predicates/CRF/rnn_1/while/Enter_3:0
!Predicates/CRF/rnn_1/while/Exit:0
#Predicates/CRF/rnn_1/while/Exit_1:0
#Predicates/CRF/rnn_1/while/Exit_2:0
#Predicates/CRF/rnn_1/while/Exit_3:0
+Predicates/CRF/rnn_1/while/ExpandDims/dim:0
'Predicates/CRF/rnn_1/while/ExpandDims:0
%Predicates/CRF/rnn_1/while/GatherNd:0
/Predicates/CRF/rnn_1/while/GreaterEqual/Enter:0
)Predicates/CRF/rnn_1/while/GreaterEqual:0
%Predicates/CRF/rnn_1/while/Identity:0
'Predicates/CRF/rnn_1/while/Identity_1:0
'Predicates/CRF/rnn_1/while/Identity_2:0
'Predicates/CRF/rnn_1/while/Identity_3:0
'Predicates/CRF/rnn_1/while/Less/Enter:0
!Predicates/CRF/rnn_1/while/Less:0
)Predicates/CRF/rnn_1/while/Less_1/Enter:0
#Predicates/CRF/rnn_1/while/Less_1:0
'Predicates/CRF/rnn_1/while/LogicalAnd:0
%Predicates/CRF/rnn_1/while/LoopCond:0
"Predicates/CRF/rnn_1/while/Merge:0
"Predicates/CRF/rnn_1/while/Merge:1
$Predicates/CRF/rnn_1/while/Merge_1:0
$Predicates/CRF/rnn_1/while/Merge_1:1
$Predicates/CRF/rnn_1/while/Merge_2:0
$Predicates/CRF/rnn_1/while/Merge_2:1
$Predicates/CRF/rnn_1/while/Merge_3:0
$Predicates/CRF/rnn_1/while/Merge_3:1
*Predicates/CRF/rnn_1/while/NextIteration:0
,Predicates/CRF/rnn_1/while/NextIteration_1:0
,Predicates/CRF/rnn_1/while/NextIteration_2:0
,Predicates/CRF/rnn_1/while/NextIteration_3:0
)Predicates/CRF/rnn_1/while/Select/Enter:0
#Predicates/CRF/rnn_1/while/Select:0
%Predicates/CRF/rnn_1/while/Select_1:0
"Predicates/CRF/rnn_1/while/Shape:0
$Predicates/CRF/rnn_1/while/Squeeze:0
#Predicates/CRF/rnn_1/while/Switch:0
#Predicates/CRF/rnn_1/while/Switch:1
%Predicates/CRF/rnn_1/while/Switch_1:0
%Predicates/CRF/rnn_1/while/Switch_1:1
%Predicates/CRF/rnn_1/while/Switch_2:0
%Predicates/CRF/rnn_1/while/Switch_2:1
%Predicates/CRF/rnn_1/while/Switch_3:0
%Predicates/CRF/rnn_1/while/Switch_3:1
4Predicates/CRF/rnn_1/while/TensorArrayReadV3/Enter:0
6Predicates/CRF/rnn_1/while/TensorArrayReadV3/Enter_1:0
.Predicates/CRF/rnn_1/while/TensorArrayReadV3:0
FPredicates/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
@Predicates/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3:0
"Predicates/CRF/rnn_1/while/add/y:0
 Predicates/CRF/rnn_1/while/add:0
$Predicates/CRF/rnn_1/while/add_1/y:0
"Predicates/CRF/rnn_1/while/add_1:0
(Predicates/CRF/rnn_1/while/range/delta:0
(Predicates/CRF/rnn_1/while/range/start:0
"Predicates/CRF/rnn_1/while/range:0
"Predicates/CRF/rnn_1/while/stack:0
0Predicates/CRF/rnn_1/while/strided_slice/stack:0
2Predicates/CRF/rnn_1/while/strided_slice/stack_1:0
2Predicates/CRF/rnn_1/while/strided_slice/stack_2:0
*Predicates/CRF/rnn_1/while/strided_slice:0
Predicates/CRF/rnn_1/zeros:0\
$Predicates/CRF/rnn_1/TensorArray_1:04Predicates/CRF/rnn_1/while/TensorArrayReadV3/Enter:0K
Predicates/CRF/rnn_1/Minimum:0)Predicates/CRF/rnn_1/while/Less_1/Enter:0l
"Predicates/CRF/rnn_1/TensorArray:0FPredicates/CRF/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Q
&Predicates/CRF/rnn_1/strided_slice_1:0'Predicates/CRF/rnn_1/while/Less/Enter:0I
Predicates/CRF/rnn_1/zeros:0)Predicates/CRF/rnn_1/while/Select/Enter:0?
QPredicates/CRF/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:06Predicates/CRF/rnn_1/while/TensorArrayReadV3/Enter_1:0U
"Predicates/CRF/rnn_1/CheckSeqLen:0/Predicates/CRF/rnn_1/while/GreaterEqual/Enter:0R"Predicates/CRF/rnn_1/while/Enter:0R$Predicates/CRF/rnn_1/while/Enter_1:0R$Predicates/CRF/rnn_1/while/Enter_2:0R$Predicates/CRF/rnn_1/while/Enter_3:0Z&Predicates/CRF/rnn_1/strided_slice_1:0
? 
(Predicates/CRF_2/rnn/while/while_context *%Predicates/CRF_2/rnn/while/LoopCond:02"Predicates/CRF_2/rnn/while/Merge:0:%Predicates/CRF_2/rnn/while/Identity:0B!Predicates/CRF_2/rnn/while/Exit:0B#Predicates/CRF_2/rnn/while/Exit_1:0B#Predicates/CRF_2/rnn/while/Exit_2:0B#Predicates/CRF_2/rnn/while/Exit_3:0J?
Predicates/CRF_2/ExpandDims:0
"Predicates/CRF_2/rnn/CheckSeqLen:0
Predicates/CRF_2/rnn/Minimum:0
"Predicates/CRF_2/rnn/TensorArray:0
QPredicates/CRF_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
$Predicates/CRF_2/rnn/TensorArray_1:0
&Predicates/CRF_2/rnn/strided_slice_1:0
-Predicates/CRF_2/rnn/while/ArgMax/dimension:0
#Predicates/CRF_2/rnn/while/ArgMax:0
!Predicates/CRF_2/rnn/while/Cast:0
"Predicates/CRF_2/rnn/while/Enter:0
$Predicates/CRF_2/rnn/while/Enter_1:0
$Predicates/CRF_2/rnn/while/Enter_2:0
$Predicates/CRF_2/rnn/while/Enter_3:0
!Predicates/CRF_2/rnn/while/Exit:0
#Predicates/CRF_2/rnn/while/Exit_1:0
#Predicates/CRF_2/rnn/while/Exit_2:0
#Predicates/CRF_2/rnn/while/Exit_3:0
+Predicates/CRF_2/rnn/while/ExpandDims/dim:0
'Predicates/CRF_2/rnn/while/ExpandDims:0
/Predicates/CRF_2/rnn/while/GreaterEqual/Enter:0
)Predicates/CRF_2/rnn/while/GreaterEqual:0
%Predicates/CRF_2/rnn/while/Identity:0
'Predicates/CRF_2/rnn/while/Identity_1:0
'Predicates/CRF_2/rnn/while/Identity_2:0
'Predicates/CRF_2/rnn/while/Identity_3:0
'Predicates/CRF_2/rnn/while/Less/Enter:0
!Predicates/CRF_2/rnn/while/Less:0
)Predicates/CRF_2/rnn/while/Less_1/Enter:0
#Predicates/CRF_2/rnn/while/Less_1:0
'Predicates/CRF_2/rnn/while/LogicalAnd:0
%Predicates/CRF_2/rnn/while/LoopCond:0
2Predicates/CRF_2/rnn/while/Max/reduction_indices:0
 Predicates/CRF_2/rnn/while/Max:0
"Predicates/CRF_2/rnn/while/Merge:0
"Predicates/CRF_2/rnn/while/Merge:1
$Predicates/CRF_2/rnn/while/Merge_1:0
$Predicates/CRF_2/rnn/while/Merge_1:1
$Predicates/CRF_2/rnn/while/Merge_2:0
$Predicates/CRF_2/rnn/while/Merge_2:1
$Predicates/CRF_2/rnn/while/Merge_3:0
$Predicates/CRF_2/rnn/while/Merge_3:1
*Predicates/CRF_2/rnn/while/NextIteration:0
,Predicates/CRF_2/rnn/while/NextIteration_1:0
,Predicates/CRF_2/rnn/while/NextIteration_2:0
,Predicates/CRF_2/rnn/while/NextIteration_3:0
)Predicates/CRF_2/rnn/while/Select/Enter:0
#Predicates/CRF_2/rnn/while/Select:0
%Predicates/CRF_2/rnn/while/Select_1:0
#Predicates/CRF_2/rnn/while/Switch:0
#Predicates/CRF_2/rnn/while/Switch:1
%Predicates/CRF_2/rnn/while/Switch_1:0
%Predicates/CRF_2/rnn/while/Switch_1:1
%Predicates/CRF_2/rnn/while/Switch_2:0
%Predicates/CRF_2/rnn/while/Switch_2:1
%Predicates/CRF_2/rnn/while/Switch_3:0
%Predicates/CRF_2/rnn/while/Switch_3:1
4Predicates/CRF_2/rnn/while/TensorArrayReadV3/Enter:0
6Predicates/CRF_2/rnn/while/TensorArrayReadV3/Enter_1:0
.Predicates/CRF_2/rnn/while/TensorArrayReadV3:0
FPredicates/CRF_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
@Predicates/CRF_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
"Predicates/CRF_2/rnn/while/add/y:0
 Predicates/CRF_2/rnn/while/add:0
(Predicates/CRF_2/rnn/while/add_1/Enter:0
"Predicates/CRF_2/rnn/while/add_1:0
"Predicates/CRF_2/rnn/while/add_2:0
$Predicates/CRF_2/rnn/while/add_3/y:0
"Predicates/CRF_2/rnn/while/add_3:0
Predicates/CRF_2/rnn/zeros:0I
Predicates/CRF_2/rnn/zeros:0)Predicates/CRF_2/rnn/while/Select/Enter:0?
QPredicates/CRF_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:06Predicates/CRF_2/rnn/while/TensorArrayReadV3/Enter_1:0l
"Predicates/CRF_2/rnn/TensorArray:0FPredicates/CRF_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0K
Predicates/CRF_2/rnn/Minimum:0)Predicates/CRF_2/rnn/while/Less_1/Enter:0U
"Predicates/CRF_2/rnn/CheckSeqLen:0/Predicates/CRF_2/rnn/while/GreaterEqual/Enter:0I
Predicates/CRF_2/ExpandDims:0(Predicates/CRF_2/rnn/while/add_1/Enter:0Q
&Predicates/CRF_2/rnn/strided_slice_1:0'Predicates/CRF_2/rnn/while/Less/Enter:0\
$Predicates/CRF_2/rnn/TensorArray_1:04Predicates/CRF_2/rnn/while/TensorArrayReadV3/Enter:0R"Predicates/CRF_2/rnn/while/Enter:0R$Predicates/CRF_2/rnn/while/Enter_1:0R$Predicates/CRF_2/rnn/while/Enter_2:0R$Predicates/CRF_2/rnn/while/Enter_3:0Z&Predicates/CRF_2/rnn/strided_slice_1:0
?!
*Predicates/CRF_2/rnn_1/while/while_context *'Predicates/CRF_2/rnn_1/while/LoopCond:02$Predicates/CRF_2/rnn_1/while/Merge:0:'Predicates/CRF_2/rnn_1/while/Identity:0B#Predicates/CRF_2/rnn_1/while/Exit:0B%Predicates/CRF_2/rnn_1/while/Exit_1:0B%Predicates/CRF_2/rnn_1/while/Exit_2:0B%Predicates/CRF_2/rnn_1/while/Exit_3:0J?
$Predicates/CRF_2/rnn_1/CheckSeqLen:0
 Predicates/CRF_2/rnn_1/Minimum:0
$Predicates/CRF_2/rnn_1/TensorArray:0
SPredicates/CRF_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
&Predicates/CRF_2/rnn_1/TensorArray_1:0
(Predicates/CRF_2/rnn_1/strided_slice_1:0
$Predicates/CRF_2/rnn_1/while/Enter:0
&Predicates/CRF_2/rnn_1/while/Enter_1:0
&Predicates/CRF_2/rnn_1/while/Enter_2:0
&Predicates/CRF_2/rnn_1/while/Enter_3:0
#Predicates/CRF_2/rnn_1/while/Exit:0
%Predicates/CRF_2/rnn_1/while/Exit_1:0
%Predicates/CRF_2/rnn_1/while/Exit_2:0
%Predicates/CRF_2/rnn_1/while/Exit_3:0
-Predicates/CRF_2/rnn_1/while/ExpandDims/dim:0
)Predicates/CRF_2/rnn_1/while/ExpandDims:0
'Predicates/CRF_2/rnn_1/while/GatherNd:0
1Predicates/CRF_2/rnn_1/while/GreaterEqual/Enter:0
+Predicates/CRF_2/rnn_1/while/GreaterEqual:0
'Predicates/CRF_2/rnn_1/while/Identity:0
)Predicates/CRF_2/rnn_1/while/Identity_1:0
)Predicates/CRF_2/rnn_1/while/Identity_2:0
)Predicates/CRF_2/rnn_1/while/Identity_3:0
)Predicates/CRF_2/rnn_1/while/Less/Enter:0
#Predicates/CRF_2/rnn_1/while/Less:0
+Predicates/CRF_2/rnn_1/while/Less_1/Enter:0
%Predicates/CRF_2/rnn_1/while/Less_1:0
)Predicates/CRF_2/rnn_1/while/LogicalAnd:0
'Predicates/CRF_2/rnn_1/while/LoopCond:0
$Predicates/CRF_2/rnn_1/while/Merge:0
$Predicates/CRF_2/rnn_1/while/Merge:1
&Predicates/CRF_2/rnn_1/while/Merge_1:0
&Predicates/CRF_2/rnn_1/while/Merge_1:1
&Predicates/CRF_2/rnn_1/while/Merge_2:0
&Predicates/CRF_2/rnn_1/while/Merge_2:1
&Predicates/CRF_2/rnn_1/while/Merge_3:0
&Predicates/CRF_2/rnn_1/while/Merge_3:1
,Predicates/CRF_2/rnn_1/while/NextIteration:0
.Predicates/CRF_2/rnn_1/while/NextIteration_1:0
.Predicates/CRF_2/rnn_1/while/NextIteration_2:0
.Predicates/CRF_2/rnn_1/while/NextIteration_3:0
+Predicates/CRF_2/rnn_1/while/Select/Enter:0
%Predicates/CRF_2/rnn_1/while/Select:0
'Predicates/CRF_2/rnn_1/while/Select_1:0
$Predicates/CRF_2/rnn_1/while/Shape:0
&Predicates/CRF_2/rnn_1/while/Squeeze:0
%Predicates/CRF_2/rnn_1/while/Switch:0
%Predicates/CRF_2/rnn_1/while/Switch:1
'Predicates/CRF_2/rnn_1/while/Switch_1:0
'Predicates/CRF_2/rnn_1/while/Switch_1:1
'Predicates/CRF_2/rnn_1/while/Switch_2:0
'Predicates/CRF_2/rnn_1/while/Switch_2:1
'Predicates/CRF_2/rnn_1/while/Switch_3:0
'Predicates/CRF_2/rnn_1/while/Switch_3:1
6Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/Enter:0
8Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/Enter_1:0
0Predicates/CRF_2/rnn_1/while/TensorArrayReadV3:0
HPredicates/CRF_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
BPredicates/CRF_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3:0
$Predicates/CRF_2/rnn_1/while/add/y:0
"Predicates/CRF_2/rnn_1/while/add:0
&Predicates/CRF_2/rnn_1/while/add_1/y:0
$Predicates/CRF_2/rnn_1/while/add_1:0
*Predicates/CRF_2/rnn_1/while/range/delta:0
*Predicates/CRF_2/rnn_1/while/range/start:0
$Predicates/CRF_2/rnn_1/while/range:0
$Predicates/CRF_2/rnn_1/while/stack:0
2Predicates/CRF_2/rnn_1/while/strided_slice/stack:0
4Predicates/CRF_2/rnn_1/while/strided_slice/stack_1:0
4Predicates/CRF_2/rnn_1/while/strided_slice/stack_2:0
,Predicates/CRF_2/rnn_1/while/strided_slice:0
Predicates/CRF_2/rnn_1/zeros:0M
Predicates/CRF_2/rnn_1/zeros:0+Predicates/CRF_2/rnn_1/while/Select/Enter:0?
SPredicates/CRF_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:08Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/Enter_1:0O
 Predicates/CRF_2/rnn_1/Minimum:0+Predicates/CRF_2/rnn_1/while/Less_1/Enter:0p
$Predicates/CRF_2/rnn_1/TensorArray:0HPredicates/CRF_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0`
&Predicates/CRF_2/rnn_1/TensorArray_1:06Predicates/CRF_2/rnn_1/while/TensorArrayReadV3/Enter:0U
(Predicates/CRF_2/rnn_1/strided_slice_1:0)Predicates/CRF_2/rnn_1/while/Less/Enter:0Y
$Predicates/CRF_2/rnn_1/CheckSeqLen:01Predicates/CRF_2/rnn_1/while/GreaterEqual/Enter:0R$Predicates/CRF_2/rnn_1/while/Enter:0R&Predicates/CRF_2/rnn_1/while/Enter_1:0R&Predicates/CRF_2/rnn_1/while/Enter_2:0R&Predicates/CRF_2/rnn_1/while/Enter_3:0Z(Predicates/CRF_2/rnn_1/strided_slice_1:0"\
table_initializerG
E
CPredicates/index_to_string/table_init/InitializeTableFromTextFileV2"%
saved_model_main_op


group_deps">
asset_filepaths+
)
'Predicates/index_to_string/asset_path:0"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"?
trainable_variables??
~
Predicates/pos/Variable:0Predicates/pos/Variable/AssignPredicates/pos/Variable/read:02Predicates/pos/random_uniform:08
~
Predicates/dep/Variable:0Predicates/dep/Variable/AssignPredicates/dep/Variable/read:02Predicates/dep/random_uniform:08
?
Predicates/mask/Variable:0Predicates/mask/Variable/AssignPredicates/mask/Variable/read:02 Predicates/mask/random_uniform:08
?
'Predicates/BiLSTM/fw/lstm_cell/kernel:0,Predicates/BiLSTM/fw/lstm_cell/kernel/Assign,Predicates/BiLSTM/fw/lstm_cell/kernel/read:02BPredicates/BiLSTM/fw/lstm_cell/kernel/Initializer/random_uniform:08
?
%Predicates/BiLSTM/fw/lstm_cell/bias:0*Predicates/BiLSTM/fw/lstm_cell/bias/Assign*Predicates/BiLSTM/fw/lstm_cell/bias/read:027Predicates/BiLSTM/fw/lstm_cell/bias/Initializer/Const:08
?
'Predicates/BiLSTM/bw/lstm_cell/kernel:0,Predicates/BiLSTM/bw/lstm_cell/kernel/Assign,Predicates/BiLSTM/bw/lstm_cell/kernel/read:02BPredicates/BiLSTM/bw/lstm_cell/kernel/Initializer/random_uniform:08
?
%Predicates/BiLSTM/bw/lstm_cell/bias:0*Predicates/BiLSTM/bw/lstm_cell/bias/Assign*Predicates/BiLSTM/bw/lstm_cell/bias/read:027Predicates/BiLSTM/bw/lstm_cell/bias/Initializer/Const:08
r
Predicates/proj/W:0Predicates/proj/W/AssignPredicates/proj/W/read:02%Predicates/proj/W/Initializer/mul_1:08
r
Predicates/proj/b:0Predicates/proj/b/AssignPredicates/proj/b/read:02%Predicates/proj/b/Initializer/zeros:08
?
Predicates/CRF/transition:0 Predicates/CRF/transition/Assign Predicates/CRF/transition/read:026Predicates/CRF/transition/Initializer/random_uniform:08
?
Predicates/pred_arc/Variable:0#Predicates/pred_arc/Variable/Assign#Predicates/pred_arc/Variable/read:02$Predicates/pred_arc/random_uniform:08
?
Predicates/proj_2/W_2:0Predicates/proj_2/W_2/AssignPredicates/proj_2/W_2/read:02)Predicates/proj_2/W_2/Initializer/mul_1:08
?
Predicates/proj_2/b_2:0Predicates/proj_2/b_2/AssignPredicates/proj_2/b_2/read:02)Predicates/proj_2/b_2/Initializer/zeros:08
?
Predicates/CRF_2/transition_2:0$Predicates/CRF_2/transition_2/Assign$Predicates/CRF_2/transition_2/read:02:Predicates/CRF_2/transition_2/Initializer/random_uniform:08"?
saved_model_assetsm*k
i
+type.googleapis.com/tensorflow.AssetFileDef:
)
'Predicates/index_to_string/asset_path:0index_tag.txt*?
serving_default?
6
pred_dep*

pred_dep:0??????????????????
8
	pred_mask+
pred_mask:0??????????????????
U
pred_original_sequence_lengths3
 pred_original_sequence_lengths:0?????????
:

pred_input,
pred_input:0??????????????????
:

labels_arc,
labels_arc:0??????????????????G
sequence_lenghts3
 pred_original_sequence_lengths:0?????????8
probabilities'
Predicates/viterbi:0?????????C
arc<
$Predicates/CRF_2/ReverseSequence_1:0??????????P
tagsH
0Predicates/hash_table_Lookup/LookupTableFindV2:0??????????A
outputs6
Predicates/Mean:0???????????????????E
classes:
"Predicates/CRF/ReverseSequence_1:0??????????tensorflow/serving/predict*?
predict?
:

pred_input,
pred_input:0??????????????????
8
	pred_mask+
pred_mask:0??????????????????
6
pred_dep*

pred_dep:0??????????????????
U
pred_original_sequence_lengths3
 pred_original_sequence_lengths:0?????????
:

labels_arc,
labels_arc:0??????????????????P
tagsH
0Predicates/hash_table_Lookup/LookupTableFindV2:0??????????C
arc<
$Predicates/CRF_2/ReverseSequence_1:0??????????A
outputs6
Predicates/Mean:0???????????????????8
probabilities'
Predicates/viterbi:0?????????E
classes:
"Predicates/CRF/ReverseSequence_1:0??????????G
sequence_lenghts3
 pred_original_sequence_lengths:0?????????tensorflow/serving/predict