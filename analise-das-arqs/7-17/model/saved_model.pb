®
®
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ó

z
dense_79/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_79/kernel
s
#dense_79/kernel/Read/ReadVariableOpReadVariableOpdense_79/kernel*
_output_shapes

:*
dtype0
r
dense_79/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_79/bias
k
!dense_79/bias/Read/ReadVariableOpReadVariableOpdense_79/bias*
_output_shapes
:*
dtype0
z
dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_80/kernel
s
#dense_80/kernel/Read/ReadVariableOpReadVariableOpdense_80/kernel*
_output_shapes

:*
dtype0
r
dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_80/bias
k
!dense_80/bias/Read/ReadVariableOpReadVariableOpdense_80/bias*
_output_shapes
:*
dtype0
z
dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_81/kernel
s
#dense_81/kernel/Read/ReadVariableOpReadVariableOpdense_81/kernel*
_output_shapes

:*
dtype0
r
dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_81/bias
k
!dense_81/bias/Read/ReadVariableOpReadVariableOpdense_81/bias*
_output_shapes
:*
dtype0
z
dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_82/kernel
s
#dense_82/kernel/Read/ReadVariableOpReadVariableOpdense_82/kernel*
_output_shapes

:*
dtype0
r
dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_82/bias
k
!dense_82/bias/Read/ReadVariableOpReadVariableOpdense_82/bias*
_output_shapes
:*
dtype0
z
dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_83/kernel
s
#dense_83/kernel/Read/ReadVariableOpReadVariableOpdense_83/kernel*
_output_shapes

:*
dtype0
r
dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_83/bias
k
!dense_83/bias/Read/ReadVariableOpReadVariableOpdense_83/bias*
_output_shapes
:*
dtype0
z
dense_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_84/kernel
s
#dense_84/kernel/Read/ReadVariableOpReadVariableOpdense_84/kernel*
_output_shapes

:*
dtype0
r
dense_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_84/bias
k
!dense_84/bias/Read/ReadVariableOpReadVariableOpdense_84/bias*
_output_shapes
:*
dtype0
z
dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_85/kernel
s
#dense_85/kernel/Read/ReadVariableOpReadVariableOpdense_85/kernel*
_output_shapes

:*
dtype0
r
dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_85/bias
k
!dense_85/bias/Read/ReadVariableOpReadVariableOpdense_85/bias*
_output_shapes
:*
dtype0
z
dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_86/kernel
s
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel*
_output_shapes

:*
dtype0
r
dense_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_86/bias
k
!dense_86/bias/Read/ReadVariableOpReadVariableOpdense_86/bias*
_output_shapes
:*
dtype0
z
dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_87/kernel
s
#dense_87/kernel/Read/ReadVariableOpReadVariableOpdense_87/kernel*
_output_shapes

:*
dtype0
r
dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_87/bias
k
!dense_87/bias/Read/ReadVariableOpReadVariableOpdense_87/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_79/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_79/kernel/m

*Adam/dense_79/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_79/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_79/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_79/bias/m
y
(Adam/dense_79/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_79/bias/m*
_output_shapes
:*
dtype0

Adam/dense_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_80/kernel/m

*Adam/dense_80/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_80/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_80/bias/m
y
(Adam/dense_80/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_80/bias/m*
_output_shapes
:*
dtype0

Adam/dense_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_81/kernel/m

*Adam/dense_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_81/bias/m
y
(Adam/dense_81/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/m*
_output_shapes
:*
dtype0

Adam/dense_82/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/m

*Adam/dense_82/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_82/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/m
y
(Adam/dense_82/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/m*
_output_shapes
:*
dtype0

Adam/dense_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_83/kernel/m

*Adam/dense_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/m
y
(Adam/dense_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/m*
_output_shapes
:*
dtype0

Adam/dense_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_84/kernel/m

*Adam/dense_84/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_84/bias/m
y
(Adam/dense_84/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/m*
_output_shapes
:*
dtype0

Adam/dense_85/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_85/kernel/m

*Adam/dense_85/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_85/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_85/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_85/bias/m
y
(Adam/dense_85/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_85/bias/m*
_output_shapes
:*
dtype0

Adam/dense_86/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_86/kernel/m

*Adam/dense_86/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_86/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_86/bias/m
y
(Adam/dense_86/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_86/bias/m*
_output_shapes
:*
dtype0

Adam/dense_87/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_87/kernel/m

*Adam/dense_87/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_87/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_87/bias/m
y
(Adam/dense_87/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/m*
_output_shapes
:*
dtype0

Adam/dense_79/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_79/kernel/v

*Adam/dense_79/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_79/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_79/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_79/bias/v
y
(Adam/dense_79/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_79/bias/v*
_output_shapes
:*
dtype0

Adam/dense_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_80/kernel/v

*Adam/dense_80/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_80/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_80/bias/v
y
(Adam/dense_80/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_80/bias/v*
_output_shapes
:*
dtype0

Adam/dense_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_81/kernel/v

*Adam/dense_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_81/bias/v
y
(Adam/dense_81/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/v*
_output_shapes
:*
dtype0

Adam/dense_82/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/v

*Adam/dense_82/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_82/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/v
y
(Adam/dense_82/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/v*
_output_shapes
:*
dtype0

Adam/dense_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_83/kernel/v

*Adam/dense_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/v
y
(Adam/dense_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/v*
_output_shapes
:*
dtype0

Adam/dense_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_84/kernel/v

*Adam/dense_84/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_84/bias/v
y
(Adam/dense_84/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/v*
_output_shapes
:*
dtype0

Adam/dense_85/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_85/kernel/v

*Adam/dense_85/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_85/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_85/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_85/bias/v
y
(Adam/dense_85/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_85/bias/v*
_output_shapes
:*
dtype0

Adam/dense_86/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_86/kernel/v

*Adam/dense_86/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_86/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_86/bias/v
y
(Adam/dense_86/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_86/bias/v*
_output_shapes
:*
dtype0

Adam/dense_87/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_87/kernel/v

*Adam/dense_87/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_87/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_87/bias/v
y
(Adam/dense_87/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
X
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÆW
value¼WB¹W B²W
Ð
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
¨
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratemmmmmm"m#m(m)m.m/m4m5m:m;m@mAmvvvvvv"v#v(v)v.v/v4v 5v¡:v¢;v£@v¤Av¥

0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
@16
A17

0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
@16
A17
 
­

Klayers
Lmetrics
Mnon_trainable_variables
Nlayer_metrics
Olayer_regularization_losses
	variables
trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEdense_79/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_79/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

Players
Qnon_trainable_variables
Rlayer_metrics
regularization_losses
Slayer_regularization_losses
	variables
trainable_variables
Tmetrics
[Y
VARIABLE_VALUEdense_80/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_80/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

Ulayers
Vnon_trainable_variables
Wlayer_metrics
regularization_losses
Xlayer_regularization_losses
	variables
trainable_variables
Ymetrics
[Y
VARIABLE_VALUEdense_81/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_81/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

Zlayers
[non_trainable_variables
\layer_metrics
regularization_losses
]layer_regularization_losses
	variables
 trainable_variables
^metrics
[Y
VARIABLE_VALUEdense_82/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_82/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
­

_layers
`non_trainable_variables
alayer_metrics
$regularization_losses
blayer_regularization_losses
%	variables
&trainable_variables
cmetrics
[Y
VARIABLE_VALUEdense_83/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_83/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
­

dlayers
enon_trainable_variables
flayer_metrics
*regularization_losses
glayer_regularization_losses
+	variables
,trainable_variables
hmetrics
[Y
VARIABLE_VALUEdense_84/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_84/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
­

ilayers
jnon_trainable_variables
klayer_metrics
0regularization_losses
llayer_regularization_losses
1	variables
2trainable_variables
mmetrics
[Y
VARIABLE_VALUEdense_85/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_85/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
­

nlayers
onon_trainable_variables
player_metrics
6regularization_losses
qlayer_regularization_losses
7	variables
8trainable_variables
rmetrics
[Y
VARIABLE_VALUEdense_86/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_86/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
­

slayers
tnon_trainable_variables
ulayer_metrics
<regularization_losses
vlayer_regularization_losses
=	variables
>trainable_variables
wmetrics
[Y
VARIABLE_VALUEdense_87/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_87/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
­

xlayers
ynon_trainable_variables
zlayer_metrics
Bregularization_losses
{layer_regularization_losses
C	variables
Dtrainable_variables
|metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

}0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
6
	~total
	count
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

~0
1

	variables
~|
VARIABLE_VALUEAdam/dense_79/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_79/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_80/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_80/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_81/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_81/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_82/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_82/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_83/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_83/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_84/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_84/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_85/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_85/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_86/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_86/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_87/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_87/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_79/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_79/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_80/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_80/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_81/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_81/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_82/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_82/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_83/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_83/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_84/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_84/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_85/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_85/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_86/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_86/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_87/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_87/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_dense_79_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_79_inputdense_79/kerneldense_79/biasdense_80/kerneldense_80/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/biasdense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/biasdense_87/kerneldense_87/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 */
f*R(
&__inference_signature_wrapper_78603590
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ç
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_79/kernel/Read/ReadVariableOp!dense_79/bias/Read/ReadVariableOp#dense_80/kernel/Read/ReadVariableOp!dense_80/bias/Read/ReadVariableOp#dense_81/kernel/Read/ReadVariableOp!dense_81/bias/Read/ReadVariableOp#dense_82/kernel/Read/ReadVariableOp!dense_82/bias/Read/ReadVariableOp#dense_83/kernel/Read/ReadVariableOp!dense_83/bias/Read/ReadVariableOp#dense_84/kernel/Read/ReadVariableOp!dense_84/bias/Read/ReadVariableOp#dense_85/kernel/Read/ReadVariableOp!dense_85/bias/Read/ReadVariableOp#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOp#dense_87/kernel/Read/ReadVariableOp!dense_87/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_79/kernel/m/Read/ReadVariableOp(Adam/dense_79/bias/m/Read/ReadVariableOp*Adam/dense_80/kernel/m/Read/ReadVariableOp(Adam/dense_80/bias/m/Read/ReadVariableOp*Adam/dense_81/kernel/m/Read/ReadVariableOp(Adam/dense_81/bias/m/Read/ReadVariableOp*Adam/dense_82/kernel/m/Read/ReadVariableOp(Adam/dense_82/bias/m/Read/ReadVariableOp*Adam/dense_83/kernel/m/Read/ReadVariableOp(Adam/dense_83/bias/m/Read/ReadVariableOp*Adam/dense_84/kernel/m/Read/ReadVariableOp(Adam/dense_84/bias/m/Read/ReadVariableOp*Adam/dense_85/kernel/m/Read/ReadVariableOp(Adam/dense_85/bias/m/Read/ReadVariableOp*Adam/dense_86/kernel/m/Read/ReadVariableOp(Adam/dense_86/bias/m/Read/ReadVariableOp*Adam/dense_87/kernel/m/Read/ReadVariableOp(Adam/dense_87/bias/m/Read/ReadVariableOp*Adam/dense_79/kernel/v/Read/ReadVariableOp(Adam/dense_79/bias/v/Read/ReadVariableOp*Adam/dense_80/kernel/v/Read/ReadVariableOp(Adam/dense_80/bias/v/Read/ReadVariableOp*Adam/dense_81/kernel/v/Read/ReadVariableOp(Adam/dense_81/bias/v/Read/ReadVariableOp*Adam/dense_82/kernel/v/Read/ReadVariableOp(Adam/dense_82/bias/v/Read/ReadVariableOp*Adam/dense_83/kernel/v/Read/ReadVariableOp(Adam/dense_83/bias/v/Read/ReadVariableOp*Adam/dense_84/kernel/v/Read/ReadVariableOp(Adam/dense_84/bias/v/Read/ReadVariableOp*Adam/dense_85/kernel/v/Read/ReadVariableOp(Adam/dense_85/bias/v/Read/ReadVariableOp*Adam/dense_86/kernel/v/Read/ReadVariableOp(Adam/dense_86/bias/v/Read/ReadVariableOp*Adam/dense_87/kernel/v/Read/ReadVariableOp(Adam/dense_87/bias/v/Read/ReadVariableOpConst*J
TinC
A2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 **
f%R#
!__inference__traced_save_78604189

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_79/kerneldense_79/biasdense_80/kerneldense_80/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/biasdense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/biasdense_87/kerneldense_87/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_79/kernel/mAdam/dense_79/bias/mAdam/dense_80/kernel/mAdam/dense_80/bias/mAdam/dense_81/kernel/mAdam/dense_81/bias/mAdam/dense_82/kernel/mAdam/dense_82/bias/mAdam/dense_83/kernel/mAdam/dense_83/bias/mAdam/dense_84/kernel/mAdam/dense_84/bias/mAdam/dense_85/kernel/mAdam/dense_85/bias/mAdam/dense_86/kernel/mAdam/dense_86/bias/mAdam/dense_87/kernel/mAdam/dense_87/bias/mAdam/dense_79/kernel/vAdam/dense_79/bias/vAdam/dense_80/kernel/vAdam/dense_80/bias/vAdam/dense_81/kernel/vAdam/dense_81/bias/vAdam/dense_82/kernel/vAdam/dense_82/bias/vAdam/dense_83/kernel/vAdam/dense_83/bias/vAdam/dense_84/kernel/vAdam/dense_84/bias/vAdam/dense_85/kernel/vAdam/dense_85/bias/vAdam/dense_86/kernel/vAdam/dense_86/bias/vAdam/dense_87/kernel/vAdam/dense_87/bias/v*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *-
f(R&
$__inference__traced_restore_78604382ü
ð	
ß
F__inference_dense_80_layer_call_and_return_conditional_losses_78603104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Âm
Ð
#__inference__wrapped_model_78603062
dense_79_input9
5sequential_10_dense_79_matmul_readvariableop_resource:
6sequential_10_dense_79_biasadd_readvariableop_resource9
5sequential_10_dense_80_matmul_readvariableop_resource:
6sequential_10_dense_80_biasadd_readvariableop_resource9
5sequential_10_dense_81_matmul_readvariableop_resource:
6sequential_10_dense_81_biasadd_readvariableop_resource9
5sequential_10_dense_82_matmul_readvariableop_resource:
6sequential_10_dense_82_biasadd_readvariableop_resource9
5sequential_10_dense_83_matmul_readvariableop_resource:
6sequential_10_dense_83_biasadd_readvariableop_resource9
5sequential_10_dense_84_matmul_readvariableop_resource:
6sequential_10_dense_84_biasadd_readvariableop_resource9
5sequential_10_dense_85_matmul_readvariableop_resource:
6sequential_10_dense_85_biasadd_readvariableop_resource9
5sequential_10_dense_86_matmul_readvariableop_resource:
6sequential_10_dense_86_biasadd_readvariableop_resource9
5sequential_10_dense_87_matmul_readvariableop_resource:
6sequential_10_dense_87_biasadd_readvariableop_resource
identity¢-sequential_10/dense_79/BiasAdd/ReadVariableOp¢,sequential_10/dense_79/MatMul/ReadVariableOp¢-sequential_10/dense_80/BiasAdd/ReadVariableOp¢,sequential_10/dense_80/MatMul/ReadVariableOp¢-sequential_10/dense_81/BiasAdd/ReadVariableOp¢,sequential_10/dense_81/MatMul/ReadVariableOp¢-sequential_10/dense_82/BiasAdd/ReadVariableOp¢,sequential_10/dense_82/MatMul/ReadVariableOp¢-sequential_10/dense_83/BiasAdd/ReadVariableOp¢,sequential_10/dense_83/MatMul/ReadVariableOp¢-sequential_10/dense_84/BiasAdd/ReadVariableOp¢,sequential_10/dense_84/MatMul/ReadVariableOp¢-sequential_10/dense_85/BiasAdd/ReadVariableOp¢,sequential_10/dense_85/MatMul/ReadVariableOp¢-sequential_10/dense_86/BiasAdd/ReadVariableOp¢,sequential_10/dense_86/MatMul/ReadVariableOp¢-sequential_10/dense_87/BiasAdd/ReadVariableOp¢,sequential_10/dense_87/MatMul/ReadVariableOpÒ
,sequential_10/dense_79/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_79_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_79/MatMul/ReadVariableOpÀ
sequential_10/dense_79/MatMulMatMuldense_79_input4sequential_10/dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_79/MatMulÑ
-sequential_10/dense_79/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_79/BiasAdd/ReadVariableOpÝ
sequential_10/dense_79/BiasAddBiasAdd'sequential_10/dense_79/MatMul:product:05sequential_10/dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_79/BiasAdd
sequential_10/dense_79/ReluRelu'sequential_10/dense_79/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_79/ReluÒ
,sequential_10/dense_80/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_80/MatMul/ReadVariableOpÛ
sequential_10/dense_80/MatMulMatMul)sequential_10/dense_79/Relu:activations:04sequential_10/dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_80/MatMulÑ
-sequential_10/dense_80/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_80/BiasAdd/ReadVariableOpÝ
sequential_10/dense_80/BiasAddBiasAdd'sequential_10/dense_80/MatMul:product:05sequential_10/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_80/BiasAdd
sequential_10/dense_80/ReluRelu'sequential_10/dense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_80/ReluÒ
,sequential_10/dense_81/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_81/MatMul/ReadVariableOpÛ
sequential_10/dense_81/MatMulMatMul)sequential_10/dense_80/Relu:activations:04sequential_10/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_81/MatMulÑ
-sequential_10/dense_81/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_81/BiasAdd/ReadVariableOpÝ
sequential_10/dense_81/BiasAddBiasAdd'sequential_10/dense_81/MatMul:product:05sequential_10/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_81/BiasAdd
sequential_10/dense_81/ReluRelu'sequential_10/dense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_81/ReluÒ
,sequential_10/dense_82/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_82/MatMul/ReadVariableOpÛ
sequential_10/dense_82/MatMulMatMul)sequential_10/dense_81/Relu:activations:04sequential_10/dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_82/MatMulÑ
-sequential_10/dense_82/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_82/BiasAdd/ReadVariableOpÝ
sequential_10/dense_82/BiasAddBiasAdd'sequential_10/dense_82/MatMul:product:05sequential_10/dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_82/BiasAdd
sequential_10/dense_82/ReluRelu'sequential_10/dense_82/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_82/ReluÒ
,sequential_10/dense_83/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_83/MatMul/ReadVariableOpÛ
sequential_10/dense_83/MatMulMatMul)sequential_10/dense_82/Relu:activations:04sequential_10/dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_83/MatMulÑ
-sequential_10/dense_83/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_83/BiasAdd/ReadVariableOpÝ
sequential_10/dense_83/BiasAddBiasAdd'sequential_10/dense_83/MatMul:product:05sequential_10/dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_83/BiasAdd
sequential_10/dense_83/ReluRelu'sequential_10/dense_83/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_83/ReluÒ
,sequential_10/dense_84/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_84_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_84/MatMul/ReadVariableOpÛ
sequential_10/dense_84/MatMulMatMul)sequential_10/dense_83/Relu:activations:04sequential_10/dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_84/MatMulÑ
-sequential_10/dense_84/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_84/BiasAdd/ReadVariableOpÝ
sequential_10/dense_84/BiasAddBiasAdd'sequential_10/dense_84/MatMul:product:05sequential_10/dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_84/BiasAdd
sequential_10/dense_84/ReluRelu'sequential_10/dense_84/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_84/ReluÒ
,sequential_10/dense_85/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_85_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_85/MatMul/ReadVariableOpÛ
sequential_10/dense_85/MatMulMatMul)sequential_10/dense_84/Relu:activations:04sequential_10/dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_85/MatMulÑ
-sequential_10/dense_85/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_85/BiasAdd/ReadVariableOpÝ
sequential_10/dense_85/BiasAddBiasAdd'sequential_10/dense_85/MatMul:product:05sequential_10/dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_85/BiasAdd
sequential_10/dense_85/ReluRelu'sequential_10/dense_85/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_85/ReluÒ
,sequential_10/dense_86/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_86/MatMul/ReadVariableOpÛ
sequential_10/dense_86/MatMulMatMul)sequential_10/dense_85/Relu:activations:04sequential_10/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_86/MatMulÑ
-sequential_10/dense_86/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_86/BiasAdd/ReadVariableOpÝ
sequential_10/dense_86/BiasAddBiasAdd'sequential_10/dense_86/MatMul:product:05sequential_10/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_86/BiasAdd
sequential_10/dense_86/ReluRelu'sequential_10/dense_86/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_86/ReluÒ
,sequential_10/dense_87/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_10/dense_87/MatMul/ReadVariableOpÛ
sequential_10/dense_87/MatMulMatMul)sequential_10/dense_86/Relu:activations:04sequential_10/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/dense_87/MatMulÑ
-sequential_10/dense_87/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_87/BiasAdd/ReadVariableOpÝ
sequential_10/dense_87/BiasAddBiasAdd'sequential_10/dense_87/MatMul:product:05sequential_10/dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_87/BiasAddÒ
IdentityIdentity'sequential_10/dense_87/BiasAdd:output:0.^sequential_10/dense_79/BiasAdd/ReadVariableOp-^sequential_10/dense_79/MatMul/ReadVariableOp.^sequential_10/dense_80/BiasAdd/ReadVariableOp-^sequential_10/dense_80/MatMul/ReadVariableOp.^sequential_10/dense_81/BiasAdd/ReadVariableOp-^sequential_10/dense_81/MatMul/ReadVariableOp.^sequential_10/dense_82/BiasAdd/ReadVariableOp-^sequential_10/dense_82/MatMul/ReadVariableOp.^sequential_10/dense_83/BiasAdd/ReadVariableOp-^sequential_10/dense_83/MatMul/ReadVariableOp.^sequential_10/dense_84/BiasAdd/ReadVariableOp-^sequential_10/dense_84/MatMul/ReadVariableOp.^sequential_10/dense_85/BiasAdd/ReadVariableOp-^sequential_10/dense_85/MatMul/ReadVariableOp.^sequential_10/dense_86/BiasAdd/ReadVariableOp-^sequential_10/dense_86/MatMul/ReadVariableOp.^sequential_10/dense_87/BiasAdd/ReadVariableOp-^sequential_10/dense_87/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2^
-sequential_10/dense_79/BiasAdd/ReadVariableOp-sequential_10/dense_79/BiasAdd/ReadVariableOp2\
,sequential_10/dense_79/MatMul/ReadVariableOp,sequential_10/dense_79/MatMul/ReadVariableOp2^
-sequential_10/dense_80/BiasAdd/ReadVariableOp-sequential_10/dense_80/BiasAdd/ReadVariableOp2\
,sequential_10/dense_80/MatMul/ReadVariableOp,sequential_10/dense_80/MatMul/ReadVariableOp2^
-sequential_10/dense_81/BiasAdd/ReadVariableOp-sequential_10/dense_81/BiasAdd/ReadVariableOp2\
,sequential_10/dense_81/MatMul/ReadVariableOp,sequential_10/dense_81/MatMul/ReadVariableOp2^
-sequential_10/dense_82/BiasAdd/ReadVariableOp-sequential_10/dense_82/BiasAdd/ReadVariableOp2\
,sequential_10/dense_82/MatMul/ReadVariableOp,sequential_10/dense_82/MatMul/ReadVariableOp2^
-sequential_10/dense_83/BiasAdd/ReadVariableOp-sequential_10/dense_83/BiasAdd/ReadVariableOp2\
,sequential_10/dense_83/MatMul/ReadVariableOp,sequential_10/dense_83/MatMul/ReadVariableOp2^
-sequential_10/dense_84/BiasAdd/ReadVariableOp-sequential_10/dense_84/BiasAdd/ReadVariableOp2\
,sequential_10/dense_84/MatMul/ReadVariableOp,sequential_10/dense_84/MatMul/ReadVariableOp2^
-sequential_10/dense_85/BiasAdd/ReadVariableOp-sequential_10/dense_85/BiasAdd/ReadVariableOp2\
,sequential_10/dense_85/MatMul/ReadVariableOp,sequential_10/dense_85/MatMul/ReadVariableOp2^
-sequential_10/dense_86/BiasAdd/ReadVariableOp-sequential_10/dense_86/BiasAdd/ReadVariableOp2\
,sequential_10/dense_86/MatMul/ReadVariableOp,sequential_10/dense_86/MatMul/ReadVariableOp2^
-sequential_10/dense_87/BiasAdd/ReadVariableOp-sequential_10/dense_87/BiasAdd/ReadVariableOp2\
,sequential_10/dense_87/MatMul/ReadVariableOp,sequential_10/dense_87/MatMul/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_79_input
ð	
ß
F__inference_dense_83_layer_call_and_return_conditional_losses_78603895

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_79_layer_call_and_return_conditional_losses_78603077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«

0__inference_sequential_10_layer_call_fn_78603449
dense_79_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCalldense_79_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *T
fORM
K__inference_sequential_10_layer_call_and_return_conditional_losses_786034102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_79_input
«

0__inference_sequential_10_layer_call_fn_78603539
dense_79_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCalldense_79_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *T
fORM
K__inference_sequential_10_layer_call_and_return_conditional_losses_786035002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_79_input
ªS
ø
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603656

inputs+
'dense_79_matmul_readvariableop_resource,
(dense_79_biasadd_readvariableop_resource+
'dense_80_matmul_readvariableop_resource,
(dense_80_biasadd_readvariableop_resource+
'dense_81_matmul_readvariableop_resource,
(dense_81_biasadd_readvariableop_resource+
'dense_82_matmul_readvariableop_resource,
(dense_82_biasadd_readvariableop_resource+
'dense_83_matmul_readvariableop_resource,
(dense_83_biasadd_readvariableop_resource+
'dense_84_matmul_readvariableop_resource,
(dense_84_biasadd_readvariableop_resource+
'dense_85_matmul_readvariableop_resource,
(dense_85_biasadd_readvariableop_resource+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource+
'dense_87_matmul_readvariableop_resource,
(dense_87_biasadd_readvariableop_resource
identity¢dense_79/BiasAdd/ReadVariableOp¢dense_79/MatMul/ReadVariableOp¢dense_80/BiasAdd/ReadVariableOp¢dense_80/MatMul/ReadVariableOp¢dense_81/BiasAdd/ReadVariableOp¢dense_81/MatMul/ReadVariableOp¢dense_82/BiasAdd/ReadVariableOp¢dense_82/MatMul/ReadVariableOp¢dense_83/BiasAdd/ReadVariableOp¢dense_83/MatMul/ReadVariableOp¢dense_84/BiasAdd/ReadVariableOp¢dense_84/MatMul/ReadVariableOp¢dense_85/BiasAdd/ReadVariableOp¢dense_85/MatMul/ReadVariableOp¢dense_86/BiasAdd/ReadVariableOp¢dense_86/MatMul/ReadVariableOp¢dense_87/BiasAdd/ReadVariableOp¢dense_87/MatMul/ReadVariableOp¨
dense_79/MatMul/ReadVariableOpReadVariableOp'dense_79_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_79/MatMul/ReadVariableOp
dense_79/MatMulMatMulinputs&dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_79/MatMul§
dense_79/BiasAdd/ReadVariableOpReadVariableOp(dense_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_79/BiasAdd/ReadVariableOp¥
dense_79/BiasAddBiasAdddense_79/MatMul:product:0'dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_79/BiasAdds
dense_79/ReluReludense_79/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_79/Relu¨
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_80/MatMul/ReadVariableOp£
dense_80/MatMulMatMuldense_79/Relu:activations:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_80/MatMul§
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_80/BiasAdd/ReadVariableOp¥
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_80/BiasAdds
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_80/Relu¨
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_81/MatMul/ReadVariableOp£
dense_81/MatMulMatMuldense_80/Relu:activations:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_81/MatMul§
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_81/BiasAdd/ReadVariableOp¥
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_81/BiasAdds
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_81/Relu¨
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_82/MatMul/ReadVariableOp£
dense_82/MatMulMatMuldense_81/Relu:activations:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_82/MatMul§
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_82/BiasAdd/ReadVariableOp¥
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_82/BiasAdds
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_82/Relu¨
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_83/MatMul/ReadVariableOp£
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_83/MatMul§
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_83/BiasAdd/ReadVariableOp¥
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_83/BiasAdds
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_83/Relu¨
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_84/MatMul/ReadVariableOp£
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_84/MatMul§
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_84/BiasAdd/ReadVariableOp¥
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_84/BiasAdds
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_84/Relu¨
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_85/MatMul/ReadVariableOp£
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_85/MatMul§
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_85/BiasAdd/ReadVariableOp¥
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_85/BiasAdds
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_85/Relu¨
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_86/MatMul/ReadVariableOp£
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_86/MatMul§
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_86/BiasAdd/ReadVariableOp¥
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_86/Relu¨
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_87/MatMul/ReadVariableOp£
dense_87/MatMulMatMuldense_86/Relu:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_87/MatMul§
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_87/BiasAdd/ReadVariableOp¥
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_87/BiasAddÈ
IdentityIdentitydense_87/BiasAdd:output:0 ^dense_79/BiasAdd/ReadVariableOp^dense_79/MatMul/ReadVariableOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2B
dense_79/BiasAdd/ReadVariableOpdense_79/BiasAdd/ReadVariableOp2@
dense_79/MatMul/ReadVariableOpdense_79/MatMul/ReadVariableOp2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_80_layer_call_and_return_conditional_losses_78603835

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö1
Ë
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603309
dense_79_input
dense_79_78603088
dense_79_78603090
dense_80_78603115
dense_80_78603117
dense_81_78603142
dense_81_78603144
dense_82_78603169
dense_82_78603171
dense_83_78603196
dense_83_78603198
dense_84_78603223
dense_84_78603225
dense_85_78603250
dense_85_78603252
dense_86_78603277
dense_86_78603279
dense_87_78603303
dense_87_78603305
identity¢ dense_79/StatefulPartitionedCall¢ dense_80/StatefulPartitionedCall¢ dense_81/StatefulPartitionedCall¢ dense_82/StatefulPartitionedCall¢ dense_83/StatefulPartitionedCall¢ dense_84/StatefulPartitionedCall¢ dense_85/StatefulPartitionedCall¢ dense_86/StatefulPartitionedCall¢ dense_87/StatefulPartitionedCall¼
 dense_79/StatefulPartitionedCallStatefulPartitionedCalldense_79_inputdense_79_78603088dense_79_78603090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_79_layer_call_and_return_conditional_losses_786030772"
 dense_79/StatefulPartitionedCall×
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_78603115dense_80_78603117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_786031042"
 dense_80/StatefulPartitionedCall×
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_78603142dense_81_78603144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_786031312"
 dense_81/StatefulPartitionedCall×
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_78603169dense_82_78603171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_786031582"
 dense_82/StatefulPartitionedCall×
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_78603196dense_83_78603198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_786031852"
 dense_83/StatefulPartitionedCall×
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_78603223dense_84_78603225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_84_layer_call_and_return_conditional_losses_786032122"
 dense_84/StatefulPartitionedCall×
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_78603250dense_85_78603252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_85_layer_call_and_return_conditional_losses_786032392"
 dense_85/StatefulPartitionedCall×
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_78603277dense_86_78603279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_86_layer_call_and_return_conditional_losses_786032662"
 dense_86/StatefulPartitionedCall×
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_78603303dense_87_78603305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_786032922"
 dense_87/StatefulPartitionedCall¸
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_79_input
û

+__inference_dense_79_layer_call_fn_78603824

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_79_layer_call_and_return_conditional_losses_786030772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
ß
F__inference_dense_87_layer_call_and_return_conditional_losses_78603292

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_86_layer_call_and_return_conditional_losses_78603266

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö1
Ë
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603358
dense_79_input
dense_79_78603312
dense_79_78603314
dense_80_78603317
dense_80_78603319
dense_81_78603322
dense_81_78603324
dense_82_78603327
dense_82_78603329
dense_83_78603332
dense_83_78603334
dense_84_78603337
dense_84_78603339
dense_85_78603342
dense_85_78603344
dense_86_78603347
dense_86_78603349
dense_87_78603352
dense_87_78603354
identity¢ dense_79/StatefulPartitionedCall¢ dense_80/StatefulPartitionedCall¢ dense_81/StatefulPartitionedCall¢ dense_82/StatefulPartitionedCall¢ dense_83/StatefulPartitionedCall¢ dense_84/StatefulPartitionedCall¢ dense_85/StatefulPartitionedCall¢ dense_86/StatefulPartitionedCall¢ dense_87/StatefulPartitionedCall¼
 dense_79/StatefulPartitionedCallStatefulPartitionedCalldense_79_inputdense_79_78603312dense_79_78603314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_79_layer_call_and_return_conditional_losses_786030772"
 dense_79/StatefulPartitionedCall×
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_78603317dense_80_78603319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_786031042"
 dense_80/StatefulPartitionedCall×
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_78603322dense_81_78603324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_786031312"
 dense_81/StatefulPartitionedCall×
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_78603327dense_82_78603329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_786031582"
 dense_82/StatefulPartitionedCall×
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_78603332dense_83_78603334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_786031852"
 dense_83/StatefulPartitionedCall×
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_78603337dense_84_78603339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_84_layer_call_and_return_conditional_losses_786032122"
 dense_84/StatefulPartitionedCall×
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_78603342dense_85_78603344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_85_layer_call_and_return_conditional_losses_786032392"
 dense_85/StatefulPartitionedCall×
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_78603347dense_86_78603349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_86_layer_call_and_return_conditional_losses_786032662"
 dense_86/StatefulPartitionedCall×
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_78603352dense_87_78603354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_786032922"
 dense_87/StatefulPartitionedCall¸
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_79_input

Ø
$__inference__traced_restore_78604382
file_prefix$
 assignvariableop_dense_79_kernel$
 assignvariableop_1_dense_79_bias&
"assignvariableop_2_dense_80_kernel$
 assignvariableop_3_dense_80_bias&
"assignvariableop_4_dense_81_kernel$
 assignvariableop_5_dense_81_bias&
"assignvariableop_6_dense_82_kernel$
 assignvariableop_7_dense_82_bias&
"assignvariableop_8_dense_83_kernel$
 assignvariableop_9_dense_83_bias'
#assignvariableop_10_dense_84_kernel%
!assignvariableop_11_dense_84_bias'
#assignvariableop_12_dense_85_kernel%
!assignvariableop_13_dense_85_bias'
#assignvariableop_14_dense_86_kernel%
!assignvariableop_15_dense_86_bias'
#assignvariableop_16_dense_87_kernel%
!assignvariableop_17_dense_87_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count.
*assignvariableop_25_adam_dense_79_kernel_m,
(assignvariableop_26_adam_dense_79_bias_m.
*assignvariableop_27_adam_dense_80_kernel_m,
(assignvariableop_28_adam_dense_80_bias_m.
*assignvariableop_29_adam_dense_81_kernel_m,
(assignvariableop_30_adam_dense_81_bias_m.
*assignvariableop_31_adam_dense_82_kernel_m,
(assignvariableop_32_adam_dense_82_bias_m.
*assignvariableop_33_adam_dense_83_kernel_m,
(assignvariableop_34_adam_dense_83_bias_m.
*assignvariableop_35_adam_dense_84_kernel_m,
(assignvariableop_36_adam_dense_84_bias_m.
*assignvariableop_37_adam_dense_85_kernel_m,
(assignvariableop_38_adam_dense_85_bias_m.
*assignvariableop_39_adam_dense_86_kernel_m,
(assignvariableop_40_adam_dense_86_bias_m.
*assignvariableop_41_adam_dense_87_kernel_m,
(assignvariableop_42_adam_dense_87_bias_m.
*assignvariableop_43_adam_dense_79_kernel_v,
(assignvariableop_44_adam_dense_79_bias_v.
*assignvariableop_45_adam_dense_80_kernel_v,
(assignvariableop_46_adam_dense_80_bias_v.
*assignvariableop_47_adam_dense_81_kernel_v,
(assignvariableop_48_adam_dense_81_bias_v.
*assignvariableop_49_adam_dense_82_kernel_v,
(assignvariableop_50_adam_dense_82_bias_v.
*assignvariableop_51_adam_dense_83_kernel_v,
(assignvariableop_52_adam_dense_83_bias_v.
*assignvariableop_53_adam_dense_84_kernel_v,
(assignvariableop_54_adam_dense_84_bias_v.
*assignvariableop_55_adam_dense_85_kernel_v,
(assignvariableop_56_adam_dense_85_bias_v.
*assignvariableop_57_adam_dense_86_kernel_v,
(assignvariableop_58_adam_dense_86_bias_v.
*assignvariableop_59_adam_dense_87_kernel_v,
(assignvariableop_60_adam_dense_87_bias_v
identity_62¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*"
value"B">B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesû
ø::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*L
dtypesB
@2>	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_79_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_79_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_80_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_80_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_81_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_81_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_82_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_82_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_83_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_83_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_84_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_84_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_85_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_85_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_86_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_86_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_87_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_87_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18¥
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19§
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¦
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22®
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¡
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¡
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_79_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_79_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_80_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_80_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_81_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_81_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_82_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_82_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_83_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_83_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_84_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_84_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_85_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_85_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_86_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_86_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41²
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_87_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42°
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_87_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_79_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_79_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45²
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_80_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46°
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_80_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47²
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_81_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48°
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_81_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49²
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_82_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50°
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_82_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51²
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_83_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52°
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_83_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53²
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_84_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54°
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_84_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_85_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56°
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_85_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57²
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_86_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58°
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_86_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59²
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_87_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60°
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_87_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_609
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_61
Identity_62IdentityIdentity_61:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_62"#
identity_62Identity_62:output:0*
_input_shapesù
ö: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ð	
ß
F__inference_dense_85_layer_call_and_return_conditional_losses_78603935

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_83_layer_call_and_return_conditional_losses_78603185

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

ú
&__inference_signature_wrapper_78603590
dense_79_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCalldense_79_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *,
f'R%
#__inference__wrapped_model_786030622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_79_input
ð	
ß
F__inference_dense_79_layer_call_and_return_conditional_losses_78603815

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_81_layer_call_and_return_conditional_losses_78603131

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_dense_86_layer_call_fn_78603964

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_86_layer_call_and_return_conditional_losses_786032662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_dense_87_layer_call_fn_78603983

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_786032922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_84_layer_call_and_return_conditional_losses_78603212

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_84_layer_call_and_return_conditional_losses_78603915

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
ß
F__inference_dense_87_layer_call_and_return_conditional_losses_78603974

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_dense_83_layer_call_fn_78603904

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_786031852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_dense_85_layer_call_fn_78603944

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_85_layer_call_and_return_conditional_losses_786032392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ëx

!__inference__traced_save_78604189
file_prefix.
*savev2_dense_79_kernel_read_readvariableop,
(savev2_dense_79_bias_read_readvariableop.
*savev2_dense_80_kernel_read_readvariableop,
(savev2_dense_80_bias_read_readvariableop.
*savev2_dense_81_kernel_read_readvariableop,
(savev2_dense_81_bias_read_readvariableop.
*savev2_dense_82_kernel_read_readvariableop,
(savev2_dense_82_bias_read_readvariableop.
*savev2_dense_83_kernel_read_readvariableop,
(savev2_dense_83_bias_read_readvariableop.
*savev2_dense_84_kernel_read_readvariableop,
(savev2_dense_84_bias_read_readvariableop.
*savev2_dense_85_kernel_read_readvariableop,
(savev2_dense_85_bias_read_readvariableop.
*savev2_dense_86_kernel_read_readvariableop,
(savev2_dense_86_bias_read_readvariableop.
*savev2_dense_87_kernel_read_readvariableop,
(savev2_dense_87_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_79_kernel_m_read_readvariableop3
/savev2_adam_dense_79_bias_m_read_readvariableop5
1savev2_adam_dense_80_kernel_m_read_readvariableop3
/savev2_adam_dense_80_bias_m_read_readvariableop5
1savev2_adam_dense_81_kernel_m_read_readvariableop3
/savev2_adam_dense_81_bias_m_read_readvariableop5
1savev2_adam_dense_82_kernel_m_read_readvariableop3
/savev2_adam_dense_82_bias_m_read_readvariableop5
1savev2_adam_dense_83_kernel_m_read_readvariableop3
/savev2_adam_dense_83_bias_m_read_readvariableop5
1savev2_adam_dense_84_kernel_m_read_readvariableop3
/savev2_adam_dense_84_bias_m_read_readvariableop5
1savev2_adam_dense_85_kernel_m_read_readvariableop3
/savev2_adam_dense_85_bias_m_read_readvariableop5
1savev2_adam_dense_86_kernel_m_read_readvariableop3
/savev2_adam_dense_86_bias_m_read_readvariableop5
1savev2_adam_dense_87_kernel_m_read_readvariableop3
/savev2_adam_dense_87_bias_m_read_readvariableop5
1savev2_adam_dense_79_kernel_v_read_readvariableop3
/savev2_adam_dense_79_bias_v_read_readvariableop5
1savev2_adam_dense_80_kernel_v_read_readvariableop3
/savev2_adam_dense_80_bias_v_read_readvariableop5
1savev2_adam_dense_81_kernel_v_read_readvariableop3
/savev2_adam_dense_81_bias_v_read_readvariableop5
1savev2_adam_dense_82_kernel_v_read_readvariableop3
/savev2_adam_dense_82_bias_v_read_readvariableop5
1savev2_adam_dense_83_kernel_v_read_readvariableop3
/savev2_adam_dense_83_bias_v_read_readvariableop5
1savev2_adam_dense_84_kernel_v_read_readvariableop3
/savev2_adam_dense_84_bias_v_read_readvariableop5
1savev2_adam_dense_85_kernel_v_read_readvariableop3
/savev2_adam_dense_85_bias_v_read_readvariableop5
1savev2_adam_dense_86_kernel_v_read_readvariableop3
/savev2_adam_dense_86_bias_v_read_readvariableop5
1savev2_adam_dense_87_kernel_v_read_readvariableop3
/savev2_adam_dense_87_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameü"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*"
value"B">B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_79_kernel_read_readvariableop(savev2_dense_79_bias_read_readvariableop*savev2_dense_80_kernel_read_readvariableop(savev2_dense_80_bias_read_readvariableop*savev2_dense_81_kernel_read_readvariableop(savev2_dense_81_bias_read_readvariableop*savev2_dense_82_kernel_read_readvariableop(savev2_dense_82_bias_read_readvariableop*savev2_dense_83_kernel_read_readvariableop(savev2_dense_83_bias_read_readvariableop*savev2_dense_84_kernel_read_readvariableop(savev2_dense_84_bias_read_readvariableop*savev2_dense_85_kernel_read_readvariableop(savev2_dense_85_bias_read_readvariableop*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop*savev2_dense_87_kernel_read_readvariableop(savev2_dense_87_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_79_kernel_m_read_readvariableop/savev2_adam_dense_79_bias_m_read_readvariableop1savev2_adam_dense_80_kernel_m_read_readvariableop/savev2_adam_dense_80_bias_m_read_readvariableop1savev2_adam_dense_81_kernel_m_read_readvariableop/savev2_adam_dense_81_bias_m_read_readvariableop1savev2_adam_dense_82_kernel_m_read_readvariableop/savev2_adam_dense_82_bias_m_read_readvariableop1savev2_adam_dense_83_kernel_m_read_readvariableop/savev2_adam_dense_83_bias_m_read_readvariableop1savev2_adam_dense_84_kernel_m_read_readvariableop/savev2_adam_dense_84_bias_m_read_readvariableop1savev2_adam_dense_85_kernel_m_read_readvariableop/savev2_adam_dense_85_bias_m_read_readvariableop1savev2_adam_dense_86_kernel_m_read_readvariableop/savev2_adam_dense_86_bias_m_read_readvariableop1savev2_adam_dense_87_kernel_m_read_readvariableop/savev2_adam_dense_87_bias_m_read_readvariableop1savev2_adam_dense_79_kernel_v_read_readvariableop/savev2_adam_dense_79_bias_v_read_readvariableop1savev2_adam_dense_80_kernel_v_read_readvariableop/savev2_adam_dense_80_bias_v_read_readvariableop1savev2_adam_dense_81_kernel_v_read_readvariableop/savev2_adam_dense_81_bias_v_read_readvariableop1savev2_adam_dense_82_kernel_v_read_readvariableop/savev2_adam_dense_82_bias_v_read_readvariableop1savev2_adam_dense_83_kernel_v_read_readvariableop/savev2_adam_dense_83_bias_v_read_readvariableop1savev2_adam_dense_84_kernel_v_read_readvariableop/savev2_adam_dense_84_bias_v_read_readvariableop1savev2_adam_dense_85_kernel_v_read_readvariableop/savev2_adam_dense_85_bias_v_read_readvariableop1savev2_adam_dense_86_kernel_v_read_readvariableop/savev2_adam_dense_86_bias_v_read_readvariableop1savev2_adam_dense_87_kernel_v_read_readvariableop/savev2_adam_dense_87_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *L
dtypesB
@2>	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*×
_input_shapesÅ
Â: ::::::::::::::::::: : : : : : : ::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::$< 

_output_shapes

:: =

_output_shapes
::>

_output_shapes
: 
û

+__inference_dense_81_layer_call_fn_78603864

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_786031312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_dense_80_layer_call_fn_78603844

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_786031042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_82_layer_call_and_return_conditional_losses_78603158

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_86_layer_call_and_return_conditional_losses_78603955

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_dense_82_layer_call_fn_78603884

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_786031582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ1
Ã
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603410

inputs
dense_79_78603364
dense_79_78603366
dense_80_78603369
dense_80_78603371
dense_81_78603374
dense_81_78603376
dense_82_78603379
dense_82_78603381
dense_83_78603384
dense_83_78603386
dense_84_78603389
dense_84_78603391
dense_85_78603394
dense_85_78603396
dense_86_78603399
dense_86_78603401
dense_87_78603404
dense_87_78603406
identity¢ dense_79/StatefulPartitionedCall¢ dense_80/StatefulPartitionedCall¢ dense_81/StatefulPartitionedCall¢ dense_82/StatefulPartitionedCall¢ dense_83/StatefulPartitionedCall¢ dense_84/StatefulPartitionedCall¢ dense_85/StatefulPartitionedCall¢ dense_86/StatefulPartitionedCall¢ dense_87/StatefulPartitionedCall´
 dense_79/StatefulPartitionedCallStatefulPartitionedCallinputsdense_79_78603364dense_79_78603366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_79_layer_call_and_return_conditional_losses_786030772"
 dense_79/StatefulPartitionedCall×
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_78603369dense_80_78603371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_786031042"
 dense_80/StatefulPartitionedCall×
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_78603374dense_81_78603376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_786031312"
 dense_81/StatefulPartitionedCall×
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_78603379dense_82_78603381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_786031582"
 dense_82/StatefulPartitionedCall×
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_78603384dense_83_78603386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_786031852"
 dense_83/StatefulPartitionedCall×
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_78603389dense_84_78603391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_84_layer_call_and_return_conditional_losses_786032122"
 dense_84/StatefulPartitionedCall×
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_78603394dense_85_78603396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_85_layer_call_and_return_conditional_losses_786032392"
 dense_85/StatefulPartitionedCall×
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_78603399dense_86_78603401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_86_layer_call_and_return_conditional_losses_786032662"
 dense_86/StatefulPartitionedCall×
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_78603404dense_87_78603406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_786032922"
 dense_87/StatefulPartitionedCall¸
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_82_layer_call_and_return_conditional_losses_78603875

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
0__inference_sequential_10_layer_call_fn_78603804

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *T
fORM
K__inference_sequential_10_layer_call_and_return_conditional_losses_786035002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_dense_84_layer_call_fn_78603924

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_84_layer_call_and_return_conditional_losses_786032122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_85_layer_call_and_return_conditional_losses_78603239

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ªS
ø
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603722

inputs+
'dense_79_matmul_readvariableop_resource,
(dense_79_biasadd_readvariableop_resource+
'dense_80_matmul_readvariableop_resource,
(dense_80_biasadd_readvariableop_resource+
'dense_81_matmul_readvariableop_resource,
(dense_81_biasadd_readvariableop_resource+
'dense_82_matmul_readvariableop_resource,
(dense_82_biasadd_readvariableop_resource+
'dense_83_matmul_readvariableop_resource,
(dense_83_biasadd_readvariableop_resource+
'dense_84_matmul_readvariableop_resource,
(dense_84_biasadd_readvariableop_resource+
'dense_85_matmul_readvariableop_resource,
(dense_85_biasadd_readvariableop_resource+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource+
'dense_87_matmul_readvariableop_resource,
(dense_87_biasadd_readvariableop_resource
identity¢dense_79/BiasAdd/ReadVariableOp¢dense_79/MatMul/ReadVariableOp¢dense_80/BiasAdd/ReadVariableOp¢dense_80/MatMul/ReadVariableOp¢dense_81/BiasAdd/ReadVariableOp¢dense_81/MatMul/ReadVariableOp¢dense_82/BiasAdd/ReadVariableOp¢dense_82/MatMul/ReadVariableOp¢dense_83/BiasAdd/ReadVariableOp¢dense_83/MatMul/ReadVariableOp¢dense_84/BiasAdd/ReadVariableOp¢dense_84/MatMul/ReadVariableOp¢dense_85/BiasAdd/ReadVariableOp¢dense_85/MatMul/ReadVariableOp¢dense_86/BiasAdd/ReadVariableOp¢dense_86/MatMul/ReadVariableOp¢dense_87/BiasAdd/ReadVariableOp¢dense_87/MatMul/ReadVariableOp¨
dense_79/MatMul/ReadVariableOpReadVariableOp'dense_79_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_79/MatMul/ReadVariableOp
dense_79/MatMulMatMulinputs&dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_79/MatMul§
dense_79/BiasAdd/ReadVariableOpReadVariableOp(dense_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_79/BiasAdd/ReadVariableOp¥
dense_79/BiasAddBiasAdddense_79/MatMul:product:0'dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_79/BiasAdds
dense_79/ReluReludense_79/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_79/Relu¨
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_80/MatMul/ReadVariableOp£
dense_80/MatMulMatMuldense_79/Relu:activations:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_80/MatMul§
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_80/BiasAdd/ReadVariableOp¥
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_80/BiasAdds
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_80/Relu¨
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_81/MatMul/ReadVariableOp£
dense_81/MatMulMatMuldense_80/Relu:activations:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_81/MatMul§
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_81/BiasAdd/ReadVariableOp¥
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_81/BiasAdds
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_81/Relu¨
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_82/MatMul/ReadVariableOp£
dense_82/MatMulMatMuldense_81/Relu:activations:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_82/MatMul§
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_82/BiasAdd/ReadVariableOp¥
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_82/BiasAdds
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_82/Relu¨
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_83/MatMul/ReadVariableOp£
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_83/MatMul§
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_83/BiasAdd/ReadVariableOp¥
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_83/BiasAdds
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_83/Relu¨
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_84/MatMul/ReadVariableOp£
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_84/MatMul§
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_84/BiasAdd/ReadVariableOp¥
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_84/BiasAdds
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_84/Relu¨
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_85/MatMul/ReadVariableOp£
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_85/MatMul§
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_85/BiasAdd/ReadVariableOp¥
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_85/BiasAdds
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_85/Relu¨
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_86/MatMul/ReadVariableOp£
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_86/MatMul§
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_86/BiasAdd/ReadVariableOp¥
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_86/Relu¨
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_87/MatMul/ReadVariableOp£
dense_87/MatMulMatMuldense_86/Relu:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_87/MatMul§
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_87/BiasAdd/ReadVariableOp¥
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_87/BiasAddÈ
IdentityIdentitydense_87/BiasAdd:output:0 ^dense_79/BiasAdd/ReadVariableOp^dense_79/MatMul/ReadVariableOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2B
dense_79/BiasAdd/ReadVariableOpdense_79/BiasAdd/ReadVariableOp2@
dense_79/MatMul/ReadVariableOpdense_79/MatMul/ReadVariableOp2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
0__inference_sequential_10_layer_call_fn_78603763

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *T
fORM
K__inference_sequential_10_layer_call_and_return_conditional_losses_786034102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
ß
F__inference_dense_81_layer_call_and_return_conditional_losses_78603855

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ1
Ã
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603500

inputs
dense_79_78603454
dense_79_78603456
dense_80_78603459
dense_80_78603461
dense_81_78603464
dense_81_78603466
dense_82_78603469
dense_82_78603471
dense_83_78603474
dense_83_78603476
dense_84_78603479
dense_84_78603481
dense_85_78603484
dense_85_78603486
dense_86_78603489
dense_86_78603491
dense_87_78603494
dense_87_78603496
identity¢ dense_79/StatefulPartitionedCall¢ dense_80/StatefulPartitionedCall¢ dense_81/StatefulPartitionedCall¢ dense_82/StatefulPartitionedCall¢ dense_83/StatefulPartitionedCall¢ dense_84/StatefulPartitionedCall¢ dense_85/StatefulPartitionedCall¢ dense_86/StatefulPartitionedCall¢ dense_87/StatefulPartitionedCall´
 dense_79/StatefulPartitionedCallStatefulPartitionedCallinputsdense_79_78603454dense_79_78603456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_79_layer_call_and_return_conditional_losses_786030772"
 dense_79/StatefulPartitionedCall×
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_78603459dense_80_78603461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_786031042"
 dense_80/StatefulPartitionedCall×
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_78603464dense_81_78603466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_786031312"
 dense_81/StatefulPartitionedCall×
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_78603469dense_82_78603471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_786031582"
 dense_82/StatefulPartitionedCall×
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_78603474dense_83_78603476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_786031852"
 dense_83/StatefulPartitionedCall×
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_78603479dense_84_78603481*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_84_layer_call_and_return_conditional_losses_786032122"
 dense_84/StatefulPartitionedCall×
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_78603484dense_85_78603486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_85_layer_call_and_return_conditional_losses_786032392"
 dense_85/StatefulPartitionedCall×
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_78603489dense_86_78603491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_86_layer_call_and_return_conditional_losses_786032662"
 dense_86/StatefulPartitionedCall×
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_78603494dense_87_78603496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*G
config_proto75

CPU

GPU 

XLA_CPU

XLA_GPU2J 8 *O
fJRH
F__inference_dense_87_layer_call_and_return_conditional_losses_786032922"
 dense_87/StatefulPartitionedCall¸
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¹
serving_default¥
I
dense_79_input7
 serving_default_dense_79_input:0ÿÿÿÿÿÿÿÿÿ<
dense_870
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ìª
L
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
¦_default_save_signature
§__call__
+¨&call_and_return_all_conditional_losses"ÓG
_tf_keras_sequential´G{"class_name": "Sequential", "name": "sequential_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_79_input"}}, {"class_name": "Dense", "config": {"name": "dense_79", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_82", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_83", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_84", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_85", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_79_input"}}, {"class_name": "Dense", "config": {"name": "dense_79", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_82", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_83", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_84", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_85", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
â

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"»
_tf_keras_layer¡{"class_name": "Dense", "name": "dense_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_79", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
ò

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
ô

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
­__call__
+®&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
ô

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_82", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_82", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
ô

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
±__call__
+²&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_83", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_83", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
ô

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
³__call__
+´&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_84", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_84", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
ô

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_85", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_85", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
ô

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 17, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
õ

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_87", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
»
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratemmmmmm"m#m(m)m.m/m4m5m:m;m@mAmvvvvvv"v#v(v)v.v/v4v 5v¡:v¢;v£@v¤Av¥"
	optimizer
¦
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
@16
A17"
trackable_list_wrapper
¦
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
@16
A17"
trackable_list_wrapper
 "
trackable_list_wrapper
Î

Klayers
Lmetrics
Mnon_trainable_variables
Nlayer_metrics
Olayer_regularization_losses
	variables
trainable_variables
regularization_losses
§__call__
¦_default_save_signature
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
-
»serving_default"
signature_map
!:2dense_79/kernel
:2dense_79/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

Players
Qnon_trainable_variables
Rlayer_metrics
regularization_losses
Slayer_regularization_losses
	variables
trainable_variables
Tmetrics
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
!:2dense_80/kernel
:2dense_80/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

Ulayers
Vnon_trainable_variables
Wlayer_metrics
regularization_losses
Xlayer_regularization_losses
	variables
trainable_variables
Ymetrics
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
!:2dense_81/kernel
:2dense_81/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

Zlayers
[non_trainable_variables
\layer_metrics
regularization_losses
]layer_regularization_losses
	variables
 trainable_variables
^metrics
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
!:2dense_82/kernel
:2dense_82/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
°

_layers
`non_trainable_variables
alayer_metrics
$regularization_losses
blayer_regularization_losses
%	variables
&trainable_variables
cmetrics
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
!:2dense_83/kernel
:2dense_83/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
°

dlayers
enon_trainable_variables
flayer_metrics
*regularization_losses
glayer_regularization_losses
+	variables
,trainable_variables
hmetrics
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
!:2dense_84/kernel
:2dense_84/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
°

ilayers
jnon_trainable_variables
klayer_metrics
0regularization_losses
llayer_regularization_losses
1	variables
2trainable_variables
mmetrics
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
!:2dense_85/kernel
:2dense_85/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
°

nlayers
onon_trainable_variables
player_metrics
6regularization_losses
qlayer_regularization_losses
7	variables
8trainable_variables
rmetrics
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
!:2dense_86/kernel
:2dense_86/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
°

slayers
tnon_trainable_variables
ulayer_metrics
<regularization_losses
vlayer_regularization_losses
=	variables
>trainable_variables
wmetrics
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
!:2dense_87/kernel
:2dense_87/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
°

xlayers
ynon_trainable_variables
zlayer_metrics
Bregularization_losses
{layer_regularization_losses
C	variables
Dtrainable_variables
|metrics
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
½
	~total
	count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
&:$2Adam/dense_79/kernel/m
 :2Adam/dense_79/bias/m
&:$2Adam/dense_80/kernel/m
 :2Adam/dense_80/bias/m
&:$2Adam/dense_81/kernel/m
 :2Adam/dense_81/bias/m
&:$2Adam/dense_82/kernel/m
 :2Adam/dense_82/bias/m
&:$2Adam/dense_83/kernel/m
 :2Adam/dense_83/bias/m
&:$2Adam/dense_84/kernel/m
 :2Adam/dense_84/bias/m
&:$2Adam/dense_85/kernel/m
 :2Adam/dense_85/bias/m
&:$2Adam/dense_86/kernel/m
 :2Adam/dense_86/bias/m
&:$2Adam/dense_87/kernel/m
 :2Adam/dense_87/bias/m
&:$2Adam/dense_79/kernel/v
 :2Adam/dense_79/bias/v
&:$2Adam/dense_80/kernel/v
 :2Adam/dense_80/bias/v
&:$2Adam/dense_81/kernel/v
 :2Adam/dense_81/bias/v
&:$2Adam/dense_82/kernel/v
 :2Adam/dense_82/bias/v
&:$2Adam/dense_83/kernel/v
 :2Adam/dense_83/bias/v
&:$2Adam/dense_84/kernel/v
 :2Adam/dense_84/bias/v
&:$2Adam/dense_85/kernel/v
 :2Adam/dense_85/bias/v
&:$2Adam/dense_86/kernel/v
 :2Adam/dense_86/bias/v
&:$2Adam/dense_87/kernel/v
 :2Adam/dense_87/bias/v
è2å
#__inference__wrapped_model_78603062½
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *-¢*
(%
dense_79_inputÿÿÿÿÿÿÿÿÿ
2
0__inference_sequential_10_layer_call_fn_78603804
0__inference_sequential_10_layer_call_fn_78603539
0__inference_sequential_10_layer_call_fn_78603763
0__inference_sequential_10_layer_call_fn_78603449À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603722
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603656
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603358
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603309À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_dense_79_layer_call_fn_78603824¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_79_layer_call_and_return_conditional_losses_78603815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_80_layer_call_fn_78603844¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_80_layer_call_and_return_conditional_losses_78603835¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_81_layer_call_fn_78603864¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_81_layer_call_and_return_conditional_losses_78603855¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_82_layer_call_fn_78603884¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_82_layer_call_and_return_conditional_losses_78603875¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_83_layer_call_fn_78603904¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_83_layer_call_and_return_conditional_losses_78603895¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_84_layer_call_fn_78603924¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_84_layer_call_and_return_conditional_losses_78603915¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_85_layer_call_fn_78603944¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_85_layer_call_and_return_conditional_losses_78603935¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_86_layer_call_fn_78603964¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_86_layer_call_and_return_conditional_losses_78603955¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_87_layer_call_fn_78603983¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_87_layer_call_and_return_conditional_losses_78603974¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÔBÑ
&__inference_signature_wrapper_78603590dense_79_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ª
#__inference__wrapped_model_78603062"#()./45:;@A7¢4
-¢*
(%
dense_79_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_87"
dense_87ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_79_layer_call_and_return_conditional_losses_78603815\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_79_layer_call_fn_78603824O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_80_layer_call_and_return_conditional_losses_78603835\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_80_layer_call_fn_78603844O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_81_layer_call_and_return_conditional_losses_78603855\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_81_layer_call_fn_78603864O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_82_layer_call_and_return_conditional_losses_78603875\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_82_layer_call_fn_78603884O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_83_layer_call_and_return_conditional_losses_78603895\()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_83_layer_call_fn_78603904O()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_84_layer_call_and_return_conditional_losses_78603915\.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_84_layer_call_fn_78603924O.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_85_layer_call_and_return_conditional_losses_78603935\45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_85_layer_call_fn_78603944O45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_86_layer_call_and_return_conditional_losses_78603955\:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_86_layer_call_fn_78603964O:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_87_layer_call_and_return_conditional_losses_78603974\@A/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_87_layer_call_fn_78603983O@A/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿË
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603309|"#()./45:;@A?¢<
5¢2
(%
dense_79_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603358|"#()./45:;@A?¢<
5¢2
(%
dense_79_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603656t"#()./45:;@A7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
K__inference_sequential_10_layer_call_and_return_conditional_losses_78603722t"#()./45:;@A7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
0__inference_sequential_10_layer_call_fn_78603449o"#()./45:;@A?¢<
5¢2
(%
dense_79_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ£
0__inference_sequential_10_layer_call_fn_78603539o"#()./45:;@A?¢<
5¢2
(%
dense_79_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_10_layer_call_fn_78603763g"#()./45:;@A7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_10_layer_call_fn_78603804g"#()./45:;@A7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¿
&__inference_signature_wrapper_78603590"#()./45:;@AI¢F
¢ 
?ª<
:
dense_79_input(%
dense_79_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_87"
dense_87ÿÿÿÿÿÿÿÿÿ