À®
³
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718°

z
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_65/kernel
s
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes

:*
dtype0
r
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_65/bias
k
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
_output_shapes
:*
dtype0
z
dense_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/* 
shared_namedense_66/kernel
s
#dense_66/kernel/Read/ReadVariableOpReadVariableOpdense_66/kernel*
_output_shapes

:/*
dtype0
r
dense_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_66/bias
k
!dense_66/bias/Read/ReadVariableOpReadVariableOpdense_66/bias*
_output_shapes
:/*
dtype0
z
dense_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_67/kernel
s
#dense_67/kernel/Read/ReadVariableOpReadVariableOpdense_67/kernel*
_output_shapes

://*
dtype0
r
dense_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_67/bias
k
!dense_67/bias/Read/ReadVariableOpReadVariableOpdense_67/bias*
_output_shapes
:/*
dtype0
z
dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_68/kernel
s
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel*
_output_shapes

://*
dtype0
r
dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_68/bias
k
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes
:/*
dtype0
z
dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_69/kernel
s
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel*
_output_shapes

://*
dtype0
r
dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_69/bias
k
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
_output_shapes
:/*
dtype0
z
dense_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_70/kernel
s
#dense_70/kernel/Read/ReadVariableOpReadVariableOpdense_70/kernel*
_output_shapes

://*
dtype0
r
dense_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_70/bias
k
!dense_70/bias/Read/ReadVariableOpReadVariableOpdense_70/bias*
_output_shapes
:/*
dtype0
z
dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_71/kernel
s
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel*
_output_shapes

://*
dtype0
r
dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_71/bias
k
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
_output_shapes
:/*
dtype0
z
dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/* 
shared_namedense_72/kernel
s
#dense_72/kernel/Read/ReadVariableOpReadVariableOpdense_72/kernel*
_output_shapes

:/*
dtype0
r
dense_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_72/bias
k
!dense_72/bias/Read/ReadVariableOpReadVariableOpdense_72/bias*
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
Adam/dense_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_65/kernel/m

*Adam/dense_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_65/bias/m
y
(Adam/dense_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/m*
_output_shapes
:*
dtype0

Adam/dense_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_66/kernel/m

*Adam/dense_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/m*
_output_shapes

:/*
dtype0

Adam/dense_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_66/bias/m
y
(Adam/dense_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_67/kernel/m

*Adam/dense_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/m*
_output_shapes

://*
dtype0

Adam/dense_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_67/bias/m
y
(Adam/dense_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_68/kernel/m

*Adam/dense_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/m*
_output_shapes

://*
dtype0

Adam/dense_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_68/bias/m
y
(Adam/dense_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_69/kernel/m

*Adam/dense_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/m*
_output_shapes

://*
dtype0

Adam/dense_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_69/bias/m
y
(Adam/dense_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_70/kernel/m

*Adam/dense_70/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_70/kernel/m*
_output_shapes

://*
dtype0

Adam/dense_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_70/bias/m
y
(Adam/dense_70/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_70/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_71/kernel/m

*Adam/dense_71/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/m*
_output_shapes

://*
dtype0

Adam/dense_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_71/bias/m
y
(Adam/dense_71/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_72/kernel/m

*Adam/dense_72/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/m*
_output_shapes

:/*
dtype0

Adam/dense_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_72/bias/m
y
(Adam/dense_72/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/m*
_output_shapes
:*
dtype0

Adam/dense_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_65/kernel/v

*Adam/dense_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_65/bias/v
y
(Adam/dense_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/v*
_output_shapes
:*
dtype0

Adam/dense_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_66/kernel/v

*Adam/dense_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/v*
_output_shapes

:/*
dtype0

Adam/dense_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_66/bias/v
y
(Adam/dense_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_67/kernel/v

*Adam/dense_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_67/bias/v
y
(Adam/dense_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_68/kernel/v

*Adam/dense_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_68/bias/v
y
(Adam/dense_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_69/kernel/v

*Adam/dense_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_69/bias/v
y
(Adam/dense_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_70/kernel/v

*Adam/dense_70/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_70/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_70/bias/v
y
(Adam/dense_70/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_70/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_71/kernel/v

*Adam/dense_71/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_71/bias/v
y
(Adam/dense_71/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_72/kernel/v

*Adam/dense_72/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/v*
_output_shapes

:/*
dtype0

Adam/dense_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_72/bias/v
y
(Adam/dense_72/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
O
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÍN
valueÃNBÀN B¹N
©
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
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
ö
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemvmwmxmymzm{!m|"m}'m~(m-m.m3m4m9m:mvvvvvv!v"v'v(v-v.v3v4v9v:v
 
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
­

regularization_losses
Dlayer_metrics
Enon_trainable_variables

Flayers
Gmetrics
	variables
Hlayer_regularization_losses
trainable_variables
 
[Y
VARIABLE_VALUEdense_65/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_65/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Ilayer_metrics
Jnon_trainable_variables

Klayers
Lmetrics
	variables
Mlayer_regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_66/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_66/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Nlayer_metrics
Onon_trainable_variables

Players
Qmetrics
	variables
Rlayer_regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_67/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_67/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Slayer_metrics
Tnon_trainable_variables

Ulayers
Vmetrics
	variables
Wlayer_regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_68/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_68/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
­
#regularization_losses
Xlayer_metrics
Ynon_trainable_variables

Zlayers
[metrics
$	variables
\layer_regularization_losses
%trainable_variables
[Y
VARIABLE_VALUEdense_69/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_69/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
­
)regularization_losses
]layer_metrics
^non_trainable_variables

_layers
`metrics
*	variables
alayer_regularization_losses
+trainable_variables
[Y
VARIABLE_VALUEdense_70/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_70/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
­
/regularization_losses
blayer_metrics
cnon_trainable_variables

dlayers
emetrics
0	variables
flayer_regularization_losses
1trainable_variables
[Y
VARIABLE_VALUEdense_71/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_71/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
­
5regularization_losses
glayer_metrics
hnon_trainable_variables

ilayers
jmetrics
6	variables
klayer_regularization_losses
7trainable_variables
[Y
VARIABLE_VALUEdense_72/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_72/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
­
;regularization_losses
llayer_metrics
mnon_trainable_variables

nlayers
ometrics
<	variables
player_regularization_losses
=trainable_variables
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
 
 
8
0
1
2
3
4
5
6
7

q0
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
4
	rtotal
	scount
t	variables
u	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

t	variables
~|
VARIABLE_VALUEAdam/dense_65/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_66/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_66/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_67/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_67/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_68/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_68/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_69/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_69/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_70/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_70/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_71/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_71/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_72/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_72/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_65/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_66/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_66/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_67/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_67/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_68/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_68/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_69/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_69/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_70/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_70/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_71/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_71/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_72/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_72/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_dense_65_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
è
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_65_inputdense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/biasdense_72/kerneldense_72/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 */
f*R(
&__inference_signature_wrapper_90772326
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
à
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_65/kernel/Read/ReadVariableOp!dense_65/bias/Read/ReadVariableOp#dense_66/kernel/Read/ReadVariableOp!dense_66/bias/Read/ReadVariableOp#dense_67/kernel/Read/ReadVariableOp!dense_67/bias/Read/ReadVariableOp#dense_68/kernel/Read/ReadVariableOp!dense_68/bias/Read/ReadVariableOp#dense_69/kernel/Read/ReadVariableOp!dense_69/bias/Read/ReadVariableOp#dense_70/kernel/Read/ReadVariableOp!dense_70/bias/Read/ReadVariableOp#dense_71/kernel/Read/ReadVariableOp!dense_71/bias/Read/ReadVariableOp#dense_72/kernel/Read/ReadVariableOp!dense_72/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_65/kernel/m/Read/ReadVariableOp(Adam/dense_65/bias/m/Read/ReadVariableOp*Adam/dense_66/kernel/m/Read/ReadVariableOp(Adam/dense_66/bias/m/Read/ReadVariableOp*Adam/dense_67/kernel/m/Read/ReadVariableOp(Adam/dense_67/bias/m/Read/ReadVariableOp*Adam/dense_68/kernel/m/Read/ReadVariableOp(Adam/dense_68/bias/m/Read/ReadVariableOp*Adam/dense_69/kernel/m/Read/ReadVariableOp(Adam/dense_69/bias/m/Read/ReadVariableOp*Adam/dense_70/kernel/m/Read/ReadVariableOp(Adam/dense_70/bias/m/Read/ReadVariableOp*Adam/dense_71/kernel/m/Read/ReadVariableOp(Adam/dense_71/bias/m/Read/ReadVariableOp*Adam/dense_72/kernel/m/Read/ReadVariableOp(Adam/dense_72/bias/m/Read/ReadVariableOp*Adam/dense_65/kernel/v/Read/ReadVariableOp(Adam/dense_65/bias/v/Read/ReadVariableOp*Adam/dense_66/kernel/v/Read/ReadVariableOp(Adam/dense_66/bias/v/Read/ReadVariableOp*Adam/dense_67/kernel/v/Read/ReadVariableOp(Adam/dense_67/bias/v/Read/ReadVariableOp*Adam/dense_68/kernel/v/Read/ReadVariableOp(Adam/dense_68/bias/v/Read/ReadVariableOp*Adam/dense_69/kernel/v/Read/ReadVariableOp(Adam/dense_69/bias/v/Read/ReadVariableOp*Adam/dense_70/kernel/v/Read/ReadVariableOp(Adam/dense_70/bias/v/Read/ReadVariableOp*Adam/dense_71/kernel/v/Read/ReadVariableOp(Adam/dense_71/bias/v/Read/ReadVariableOp*Adam/dense_72/kernel/v/Read/ReadVariableOp(Adam/dense_72/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 **
f%R#
!__inference__traced_save_90772865

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/biasdense_72/kerneldense_72/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_65/kernel/mAdam/dense_65/bias/mAdam/dense_66/kernel/mAdam/dense_66/bias/mAdam/dense_67/kernel/mAdam/dense_67/bias/mAdam/dense_68/kernel/mAdam/dense_68/bias/mAdam/dense_69/kernel/mAdam/dense_69/bias/mAdam/dense_70/kernel/mAdam/dense_70/bias/mAdam/dense_71/kernel/mAdam/dense_71/bias/mAdam/dense_72/kernel/mAdam/dense_72/bias/mAdam/dense_65/kernel/vAdam/dense_65/bias/vAdam/dense_66/kernel/vAdam/dense_66/bias/vAdam/dense_67/kernel/vAdam/dense_67/bias/vAdam/dense_68/kernel/vAdam/dense_68/bias/vAdam/dense_69/kernel/vAdam/dense_69/bias/vAdam/dense_70/kernel/vAdam/dense_70/bias/vAdam/dense_71/kernel/vAdam/dense_71/bias/vAdam/dense_72/kernel/vAdam/dense_72/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *-
f(R&
$__inference__traced_restore_90773040Ö¤
ò

&__inference_signature_wrapper_90772326
dense_65_input
unknown:
	unknown_0:
	unknown_1:/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5://
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCalldense_65_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *,
f'R%
#__inference__wrapped_model_907717802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_65_input
ìK

J__inference_sequential_9_layer_call_and_return_conditional_losses_90772518

inputs9
'dense_65_matmul_readvariableop_resource:6
(dense_65_biasadd_readvariableop_resource:9
'dense_66_matmul_readvariableop_resource:/6
(dense_66_biasadd_readvariableop_resource:/9
'dense_67_matmul_readvariableop_resource://6
(dense_67_biasadd_readvariableop_resource:/9
'dense_68_matmul_readvariableop_resource://6
(dense_68_biasadd_readvariableop_resource:/9
'dense_69_matmul_readvariableop_resource://6
(dense_69_biasadd_readvariableop_resource:/9
'dense_70_matmul_readvariableop_resource://6
(dense_70_biasadd_readvariableop_resource:/9
'dense_71_matmul_readvariableop_resource://6
(dense_71_biasadd_readvariableop_resource:/9
'dense_72_matmul_readvariableop_resource:/6
(dense_72_biasadd_readvariableop_resource:
identity¢dense_65/BiasAdd/ReadVariableOp¢dense_65/MatMul/ReadVariableOp¢dense_66/BiasAdd/ReadVariableOp¢dense_66/MatMul/ReadVariableOp¢dense_67/BiasAdd/ReadVariableOp¢dense_67/MatMul/ReadVariableOp¢dense_68/BiasAdd/ReadVariableOp¢dense_68/MatMul/ReadVariableOp¢dense_69/BiasAdd/ReadVariableOp¢dense_69/MatMul/ReadVariableOp¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢dense_72/BiasAdd/ReadVariableOp¢dense_72/MatMul/ReadVariableOp¨
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_65/MatMul/ReadVariableOp
dense_65/MatMulMatMulinputs&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_65/MatMul§
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_65/BiasAdd/ReadVariableOp¥
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_65/BiasAdds
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_65/Relu¨
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02 
dense_66/MatMul/ReadVariableOp£
dense_66/MatMulMatMuldense_65/Relu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_66/MatMul§
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_66/BiasAdd/ReadVariableOp¥
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_66/BiasAdds
dense_66/ReluReludense_66/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_66/Relu¨
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_67/MatMul/ReadVariableOp£
dense_67/MatMulMatMuldense_66/Relu:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_67/MatMul§
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_67/BiasAdd/ReadVariableOp¥
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_67/BiasAdds
dense_67/ReluReludense_67/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_67/Relu¨
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_68/MatMul/ReadVariableOp£
dense_68/MatMulMatMuldense_67/Relu:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_68/MatMul§
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_68/BiasAdd/ReadVariableOp¥
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_68/BiasAdds
dense_68/ReluReludense_68/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_68/Relu¨
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_69/MatMul/ReadVariableOp£
dense_69/MatMulMatMuldense_68/Relu:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_69/MatMul§
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_69/BiasAdd/ReadVariableOp¥
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_69/BiasAdds
dense_69/ReluReludense_69/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_69/Relu¨
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_70/MatMul/ReadVariableOp£
dense_70/MatMulMatMuldense_69/Relu:activations:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_70/MatMul§
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_70/BiasAdd/ReadVariableOp¥
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_70/BiasAdds
dense_70/ReluReludense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_70/Relu¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_71/MatMul/ReadVariableOp£
dense_71/MatMulMatMuldense_70/Relu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_71/BiasAdds
dense_71/ReluReludense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_71/Relu¨
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02 
dense_72/MatMul/ReadVariableOp£
dense_72/MatMulMatMuldense_71/Relu:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/MatMul§
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_72/BiasAdd/ReadVariableOp¥
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/BiasAdd
IdentityIdentitydense_72/BiasAdd:output:0 ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬

+__inference_dense_65_layer_call_fn_90772527

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
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
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_907717982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

÷
F__inference_dense_70_layer_call_and_return_conditional_losses_90771883

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_68_layer_call_and_return_conditional_losses_90772598

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¡n
Ê
!__inference__traced_save_90772865
file_prefix.
*savev2_dense_65_kernel_read_readvariableop,
(savev2_dense_65_bias_read_readvariableop.
*savev2_dense_66_kernel_read_readvariableop,
(savev2_dense_66_bias_read_readvariableop.
*savev2_dense_67_kernel_read_readvariableop,
(savev2_dense_67_bias_read_readvariableop.
*savev2_dense_68_kernel_read_readvariableop,
(savev2_dense_68_bias_read_readvariableop.
*savev2_dense_69_kernel_read_readvariableop,
(savev2_dense_69_bias_read_readvariableop.
*savev2_dense_70_kernel_read_readvariableop,
(savev2_dense_70_bias_read_readvariableop.
*savev2_dense_71_kernel_read_readvariableop,
(savev2_dense_71_bias_read_readvariableop.
*savev2_dense_72_kernel_read_readvariableop,
(savev2_dense_72_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_65_kernel_m_read_readvariableop3
/savev2_adam_dense_65_bias_m_read_readvariableop5
1savev2_adam_dense_66_kernel_m_read_readvariableop3
/savev2_adam_dense_66_bias_m_read_readvariableop5
1savev2_adam_dense_67_kernel_m_read_readvariableop3
/savev2_adam_dense_67_bias_m_read_readvariableop5
1savev2_adam_dense_68_kernel_m_read_readvariableop3
/savev2_adam_dense_68_bias_m_read_readvariableop5
1savev2_adam_dense_69_kernel_m_read_readvariableop3
/savev2_adam_dense_69_bias_m_read_readvariableop5
1savev2_adam_dense_70_kernel_m_read_readvariableop3
/savev2_adam_dense_70_bias_m_read_readvariableop5
1savev2_adam_dense_71_kernel_m_read_readvariableop3
/savev2_adam_dense_71_bias_m_read_readvariableop5
1savev2_adam_dense_72_kernel_m_read_readvariableop3
/savev2_adam_dense_72_bias_m_read_readvariableop5
1savev2_adam_dense_65_kernel_v_read_readvariableop3
/savev2_adam_dense_65_bias_v_read_readvariableop5
1savev2_adam_dense_66_kernel_v_read_readvariableop3
/savev2_adam_dense_66_bias_v_read_readvariableop5
1savev2_adam_dense_67_kernel_v_read_readvariableop3
/savev2_adam_dense_67_bias_v_read_readvariableop5
1savev2_adam_dense_68_kernel_v_read_readvariableop3
/savev2_adam_dense_68_bias_v_read_readvariableop5
1savev2_adam_dense_69_kernel_v_read_readvariableop3
/savev2_adam_dense_69_bias_v_read_readvariableop5
1savev2_adam_dense_70_kernel_v_read_readvariableop3
/savev2_adam_dense_70_bias_v_read_readvariableop5
1savev2_adam_dense_71_kernel_v_read_readvariableop3
/savev2_adam_dense_71_bias_v_read_readvariableop5
1savev2_adam_dense_72_kernel_v_read_readvariableop3
/savev2_adam_dense_72_bias_v_read_readvariableop
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
ShardedFilenameÂ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*Ô
valueÊBÇ8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesù
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesñ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_65_kernel_read_readvariableop(savev2_dense_65_bias_read_readvariableop*savev2_dense_66_kernel_read_readvariableop(savev2_dense_66_bias_read_readvariableop*savev2_dense_67_kernel_read_readvariableop(savev2_dense_67_bias_read_readvariableop*savev2_dense_68_kernel_read_readvariableop(savev2_dense_68_bias_read_readvariableop*savev2_dense_69_kernel_read_readvariableop(savev2_dense_69_bias_read_readvariableop*savev2_dense_70_kernel_read_readvariableop(savev2_dense_70_bias_read_readvariableop*savev2_dense_71_kernel_read_readvariableop(savev2_dense_71_bias_read_readvariableop*savev2_dense_72_kernel_read_readvariableop(savev2_dense_72_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_65_kernel_m_read_readvariableop/savev2_adam_dense_65_bias_m_read_readvariableop1savev2_adam_dense_66_kernel_m_read_readvariableop/savev2_adam_dense_66_bias_m_read_readvariableop1savev2_adam_dense_67_kernel_m_read_readvariableop/savev2_adam_dense_67_bias_m_read_readvariableop1savev2_adam_dense_68_kernel_m_read_readvariableop/savev2_adam_dense_68_bias_m_read_readvariableop1savev2_adam_dense_69_kernel_m_read_readvariableop/savev2_adam_dense_69_bias_m_read_readvariableop1savev2_adam_dense_70_kernel_m_read_readvariableop/savev2_adam_dense_70_bias_m_read_readvariableop1savev2_adam_dense_71_kernel_m_read_readvariableop/savev2_adam_dense_71_bias_m_read_readvariableop1savev2_adam_dense_72_kernel_m_read_readvariableop/savev2_adam_dense_72_bias_m_read_readvariableop1savev2_adam_dense_65_kernel_v_read_readvariableop/savev2_adam_dense_65_bias_v_read_readvariableop1savev2_adam_dense_66_kernel_v_read_readvariableop/savev2_adam_dense_66_bias_v_read_readvariableop1savev2_adam_dense_67_kernel_v_read_readvariableop/savev2_adam_dense_67_bias_v_read_readvariableop1savev2_adam_dense_68_kernel_v_read_readvariableop/savev2_adam_dense_68_bias_v_read_readvariableop1savev2_adam_dense_69_kernel_v_read_readvariableop/savev2_adam_dense_69_bias_v_read_readvariableop1savev2_adam_dense_70_kernel_v_read_readvariableop/savev2_adam_dense_70_bias_v_read_readvariableop1savev2_adam_dense_71_kernel_v_read_readvariableop/savev2_adam_dense_71_bias_v_read_readvariableop1savev2_adam_dense_72_kernel_v_read_readvariableop/savev2_adam_dense_72_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
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

identity_1Identity_1:output:0*§
_input_shapes
: :::/:/://:/://:/://:/://:/://:/:/:: : : : : : : :::/:/://:/://:/://:/://:/://:/:/::::/:/://:/://:/://:/://:/://:/:/:: 2(
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

:/: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$	 

_output_shapes

://: 


_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:/: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$  

_output_shapes

://: !

_output_shapes
:/:$" 

_output_shapes

://: #

_output_shapes
:/:$$ 

_output_shapes

://: %

_output_shapes
:/:$& 

_output_shapes

:/: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:/: +

_output_shapes
:/:$, 

_output_shapes

://: -

_output_shapes
:/:$. 

_output_shapes

://: /

_output_shapes
:/:$0 

_output_shapes

://: 1

_output_shapes
:/:$2 

_output_shapes

://: 3

_output_shapes
:/:$4 

_output_shapes

://: 5

_output_shapes
:/:$6 

_output_shapes

:/: 7

_output_shapes
::8

_output_shapes
: 
¬

+__inference_dense_66_layer_call_fn_90772547

inputs
unknown:/
	unknown_0:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_907718152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬

+__inference_dense_70_layer_call_fn_90772627

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_907718832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
µa

#__inference__wrapped_model_90771780
dense_65_inputF
4sequential_9_dense_65_matmul_readvariableop_resource:C
5sequential_9_dense_65_biasadd_readvariableop_resource:F
4sequential_9_dense_66_matmul_readvariableop_resource:/C
5sequential_9_dense_66_biasadd_readvariableop_resource:/F
4sequential_9_dense_67_matmul_readvariableop_resource://C
5sequential_9_dense_67_biasadd_readvariableop_resource:/F
4sequential_9_dense_68_matmul_readvariableop_resource://C
5sequential_9_dense_68_biasadd_readvariableop_resource:/F
4sequential_9_dense_69_matmul_readvariableop_resource://C
5sequential_9_dense_69_biasadd_readvariableop_resource:/F
4sequential_9_dense_70_matmul_readvariableop_resource://C
5sequential_9_dense_70_biasadd_readvariableop_resource:/F
4sequential_9_dense_71_matmul_readvariableop_resource://C
5sequential_9_dense_71_biasadd_readvariableop_resource:/F
4sequential_9_dense_72_matmul_readvariableop_resource:/C
5sequential_9_dense_72_biasadd_readvariableop_resource:
identity¢,sequential_9/dense_65/BiasAdd/ReadVariableOp¢+sequential_9/dense_65/MatMul/ReadVariableOp¢,sequential_9/dense_66/BiasAdd/ReadVariableOp¢+sequential_9/dense_66/MatMul/ReadVariableOp¢,sequential_9/dense_67/BiasAdd/ReadVariableOp¢+sequential_9/dense_67/MatMul/ReadVariableOp¢,sequential_9/dense_68/BiasAdd/ReadVariableOp¢+sequential_9/dense_68/MatMul/ReadVariableOp¢,sequential_9/dense_69/BiasAdd/ReadVariableOp¢+sequential_9/dense_69/MatMul/ReadVariableOp¢,sequential_9/dense_70/BiasAdd/ReadVariableOp¢+sequential_9/dense_70/MatMul/ReadVariableOp¢,sequential_9/dense_71/BiasAdd/ReadVariableOp¢+sequential_9/dense_71/MatMul/ReadVariableOp¢,sequential_9/dense_72/BiasAdd/ReadVariableOp¢+sequential_9/dense_72/MatMul/ReadVariableOpÏ
+sequential_9/dense_65/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_65_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_9/dense_65/MatMul/ReadVariableOp½
sequential_9/dense_65/MatMulMatMuldense_65_input3sequential_9/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/dense_65/MatMulÎ
,sequential_9/dense_65/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_65/BiasAdd/ReadVariableOpÙ
sequential_9/dense_65/BiasAddBiasAdd&sequential_9/dense_65/MatMul:product:04sequential_9/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/dense_65/BiasAdd
sequential_9/dense_65/ReluRelu&sequential_9/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/dense_65/ReluÏ
+sequential_9/dense_66/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_66_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02-
+sequential_9/dense_66/MatMul/ReadVariableOp×
sequential_9/dense_66/MatMulMatMul(sequential_9/dense_65/Relu:activations:03sequential_9/dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_66/MatMulÎ
,sequential_9/dense_66/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_66_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02.
,sequential_9/dense_66/BiasAdd/ReadVariableOpÙ
sequential_9/dense_66/BiasAddBiasAdd&sequential_9/dense_66/MatMul:product:04sequential_9/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_66/BiasAdd
sequential_9/dense_66/ReluRelu&sequential_9/dense_66/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_66/ReluÏ
+sequential_9/dense_67/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_67_matmul_readvariableop_resource*
_output_shapes

://*
dtype02-
+sequential_9/dense_67/MatMul/ReadVariableOp×
sequential_9/dense_67/MatMulMatMul(sequential_9/dense_66/Relu:activations:03sequential_9/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_67/MatMulÎ
,sequential_9/dense_67/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_67_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02.
,sequential_9/dense_67/BiasAdd/ReadVariableOpÙ
sequential_9/dense_67/BiasAddBiasAdd&sequential_9/dense_67/MatMul:product:04sequential_9/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_67/BiasAdd
sequential_9/dense_67/ReluRelu&sequential_9/dense_67/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_67/ReluÏ
+sequential_9/dense_68/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_68_matmul_readvariableop_resource*
_output_shapes

://*
dtype02-
+sequential_9/dense_68/MatMul/ReadVariableOp×
sequential_9/dense_68/MatMulMatMul(sequential_9/dense_67/Relu:activations:03sequential_9/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_68/MatMulÎ
,sequential_9/dense_68/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_68_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02.
,sequential_9/dense_68/BiasAdd/ReadVariableOpÙ
sequential_9/dense_68/BiasAddBiasAdd&sequential_9/dense_68/MatMul:product:04sequential_9/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_68/BiasAdd
sequential_9/dense_68/ReluRelu&sequential_9/dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_68/ReluÏ
+sequential_9/dense_69/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_69_matmul_readvariableop_resource*
_output_shapes

://*
dtype02-
+sequential_9/dense_69/MatMul/ReadVariableOp×
sequential_9/dense_69/MatMulMatMul(sequential_9/dense_68/Relu:activations:03sequential_9/dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_69/MatMulÎ
,sequential_9/dense_69/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_69_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02.
,sequential_9/dense_69/BiasAdd/ReadVariableOpÙ
sequential_9/dense_69/BiasAddBiasAdd&sequential_9/dense_69/MatMul:product:04sequential_9/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_69/BiasAdd
sequential_9/dense_69/ReluRelu&sequential_9/dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_69/ReluÏ
+sequential_9/dense_70/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_70_matmul_readvariableop_resource*
_output_shapes

://*
dtype02-
+sequential_9/dense_70/MatMul/ReadVariableOp×
sequential_9/dense_70/MatMulMatMul(sequential_9/dense_69/Relu:activations:03sequential_9/dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_70/MatMulÎ
,sequential_9/dense_70/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_70_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02.
,sequential_9/dense_70/BiasAdd/ReadVariableOpÙ
sequential_9/dense_70/BiasAddBiasAdd&sequential_9/dense_70/MatMul:product:04sequential_9/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_70/BiasAdd
sequential_9/dense_70/ReluRelu&sequential_9/dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_70/ReluÏ
+sequential_9/dense_71/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_71_matmul_readvariableop_resource*
_output_shapes

://*
dtype02-
+sequential_9/dense_71/MatMul/ReadVariableOp×
sequential_9/dense_71/MatMulMatMul(sequential_9/dense_70/Relu:activations:03sequential_9/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_71/MatMulÎ
,sequential_9/dense_71/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_71_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02.
,sequential_9/dense_71/BiasAdd/ReadVariableOpÙ
sequential_9/dense_71/BiasAddBiasAdd&sequential_9/dense_71/MatMul:product:04sequential_9/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_71/BiasAdd
sequential_9/dense_71/ReluRelu&sequential_9/dense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
sequential_9/dense_71/ReluÏ
+sequential_9/dense_72/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_72_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02-
+sequential_9/dense_72/MatMul/ReadVariableOp×
sequential_9/dense_72/MatMulMatMul(sequential_9/dense_71/Relu:activations:03sequential_9/dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/dense_72/MatMulÎ
,sequential_9/dense_72/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_72/BiasAdd/ReadVariableOpÙ
sequential_9/dense_72/BiasAddBiasAdd&sequential_9/dense_72/MatMul:product:04sequential_9/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/dense_72/BiasAddâ
IdentityIdentity&sequential_9/dense_72/BiasAdd:output:0-^sequential_9/dense_65/BiasAdd/ReadVariableOp,^sequential_9/dense_65/MatMul/ReadVariableOp-^sequential_9/dense_66/BiasAdd/ReadVariableOp,^sequential_9/dense_66/MatMul/ReadVariableOp-^sequential_9/dense_67/BiasAdd/ReadVariableOp,^sequential_9/dense_67/MatMul/ReadVariableOp-^sequential_9/dense_68/BiasAdd/ReadVariableOp,^sequential_9/dense_68/MatMul/ReadVariableOp-^sequential_9/dense_69/BiasAdd/ReadVariableOp,^sequential_9/dense_69/MatMul/ReadVariableOp-^sequential_9/dense_70/BiasAdd/ReadVariableOp,^sequential_9/dense_70/MatMul/ReadVariableOp-^sequential_9/dense_71/BiasAdd/ReadVariableOp,^sequential_9/dense_71/MatMul/ReadVariableOp-^sequential_9/dense_72/BiasAdd/ReadVariableOp,^sequential_9/dense_72/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,sequential_9/dense_65/BiasAdd/ReadVariableOp,sequential_9/dense_65/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_65/MatMul/ReadVariableOp+sequential_9/dense_65/MatMul/ReadVariableOp2\
,sequential_9/dense_66/BiasAdd/ReadVariableOp,sequential_9/dense_66/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_66/MatMul/ReadVariableOp+sequential_9/dense_66/MatMul/ReadVariableOp2\
,sequential_9/dense_67/BiasAdd/ReadVariableOp,sequential_9/dense_67/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_67/MatMul/ReadVariableOp+sequential_9/dense_67/MatMul/ReadVariableOp2\
,sequential_9/dense_68/BiasAdd/ReadVariableOp,sequential_9/dense_68/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_68/MatMul/ReadVariableOp+sequential_9/dense_68/MatMul/ReadVariableOp2\
,sequential_9/dense_69/BiasAdd/ReadVariableOp,sequential_9/dense_69/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_69/MatMul/ReadVariableOp+sequential_9/dense_69/MatMul/ReadVariableOp2\
,sequential_9/dense_70/BiasAdd/ReadVariableOp,sequential_9/dense_70/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_70/MatMul/ReadVariableOp+sequential_9/dense_70/MatMul/ReadVariableOp2\
,sequential_9/dense_71/BiasAdd/ReadVariableOp,sequential_9/dense_71/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_71/MatMul/ReadVariableOp+sequential_9/dense_71/MatMul/ReadVariableOp2\
,sequential_9/dense_72/BiasAdd/ReadVariableOp,sequential_9/dense_72/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_72/MatMul/ReadVariableOp+sequential_9/dense_72/MatMul/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_65_input
®

÷
F__inference_dense_68_layer_call_and_return_conditional_losses_90771849

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
£-
±
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772121

inputs#
dense_65_90772080:
dense_65_90772082:#
dense_66_90772085:/
dense_66_90772087:/#
dense_67_90772090://
dense_67_90772092:/#
dense_68_90772095://
dense_68_90772097:/#
dense_69_90772100://
dense_69_90772102:/#
dense_70_90772105://
dense_70_90772107:/#
dense_71_90772110://
dense_71_90772112:/#
dense_72_90772115:/
dense_72_90772117:
identity¢ dense_65/StatefulPartitionedCall¢ dense_66/StatefulPartitionedCall¢ dense_67/StatefulPartitionedCall¢ dense_68/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall§
 dense_65/StatefulPartitionedCallStatefulPartitionedCallinputsdense_65_90772080dense_65_90772082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_907717982"
 dense_65/StatefulPartitionedCallÊ
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_90772085dense_66_90772087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_907718152"
 dense_66/StatefulPartitionedCallÊ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_90772090dense_67_90772092*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_907718322"
 dense_67/StatefulPartitionedCallÊ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_90772095dense_68_90772097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_907718492"
 dense_68/StatefulPartitionedCallÊ
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_90772100dense_69_90772102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_907718662"
 dense_69/StatefulPartitionedCallÊ
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_90772105dense_70_90772107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_907718832"
 dense_70/StatefulPartitionedCallÊ
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_90772110dense_71_90772112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_907719002"
 dense_71/StatefulPartitionedCallÊ
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_90772115dense_72_90772117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_72_layer_call_and_return_conditional_losses_907719162"
 dense_72/StatefulPartitionedCall
IdentityIdentity)dense_72/StatefulPartitionedCall:output:0!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»-
¹
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772281
dense_65_input#
dense_65_90772240:
dense_65_90772242:#
dense_66_90772245:/
dense_66_90772247:/#
dense_67_90772250://
dense_67_90772252:/#
dense_68_90772255://
dense_68_90772257:/#
dense_69_90772260://
dense_69_90772262:/#
dense_70_90772265://
dense_70_90772267:/#
dense_71_90772270://
dense_71_90772272:/#
dense_72_90772275:/
dense_72_90772277:
identity¢ dense_65/StatefulPartitionedCall¢ dense_66/StatefulPartitionedCall¢ dense_67/StatefulPartitionedCall¢ dense_68/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall¯
 dense_65/StatefulPartitionedCallStatefulPartitionedCalldense_65_inputdense_65_90772240dense_65_90772242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_907717982"
 dense_65/StatefulPartitionedCallÊ
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_90772245dense_66_90772247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_907718152"
 dense_66/StatefulPartitionedCallÊ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_90772250dense_67_90772252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_907718322"
 dense_67/StatefulPartitionedCallÊ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_90772255dense_68_90772257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_907718492"
 dense_68/StatefulPartitionedCallÊ
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_90772260dense_69_90772262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_907718662"
 dense_69/StatefulPartitionedCallÊ
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_90772265dense_70_90772267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_907718832"
 dense_70/StatefulPartitionedCallÊ
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_90772270dense_71_90772272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_907719002"
 dense_71/StatefulPartitionedCallÊ
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_90772275dense_72_90772277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_72_layer_call_and_return_conditional_losses_907719162"
 dense_72/StatefulPartitionedCall
IdentityIdentity)dense_72/StatefulPartitionedCall:output:0!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_65_input
¬

+__inference_dense_67_layer_call_fn_90772567

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_907718322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_69_layer_call_and_return_conditional_losses_90771866

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¬

+__inference_dense_72_layer_call_fn_90772667

inputs
unknown:/
	unknown_0:
identity¢StatefulPartitionedCall
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
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_72_layer_call_and_return_conditional_losses_907719162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¢
£
/__inference_sequential_9_layer_call_fn_90772193
dense_65_input
unknown:
	unknown_0:
	unknown_1:/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5://
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCalldense_65_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_907721212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_65_input
¬

+__inference_dense_68_layer_call_fn_90772587

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_907718492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_67_layer_call_and_return_conditional_losses_90772578

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¢
£
/__inference_sequential_9_layer_call_fn_90771958
dense_65_input
unknown:
	unknown_0:
	unknown_1:/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5://
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCalldense_65_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_907719232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_65_input
®

÷
F__inference_dense_66_layer_call_and_return_conditional_losses_90772558

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò	
÷
F__inference_dense_72_layer_call_and_return_conditional_losses_90771916

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¬

+__inference_dense_69_layer_call_fn_90772607

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_907718662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
§ë
²!
$__inference__traced_restore_90773040
file_prefix2
 assignvariableop_dense_65_kernel:.
 assignvariableop_1_dense_65_bias:4
"assignvariableop_2_dense_66_kernel:/.
 assignvariableop_3_dense_66_bias:/4
"assignvariableop_4_dense_67_kernel://.
 assignvariableop_5_dense_67_bias:/4
"assignvariableop_6_dense_68_kernel://.
 assignvariableop_7_dense_68_bias:/4
"assignvariableop_8_dense_69_kernel://.
 assignvariableop_9_dense_69_bias:/5
#assignvariableop_10_dense_70_kernel:///
!assignvariableop_11_dense_70_bias:/5
#assignvariableop_12_dense_71_kernel:///
!assignvariableop_13_dense_71_bias:/5
#assignvariableop_14_dense_72_kernel://
!assignvariableop_15_dense_72_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: <
*assignvariableop_23_adam_dense_65_kernel_m:6
(assignvariableop_24_adam_dense_65_bias_m:<
*assignvariableop_25_adam_dense_66_kernel_m:/6
(assignvariableop_26_adam_dense_66_bias_m:/<
*assignvariableop_27_adam_dense_67_kernel_m://6
(assignvariableop_28_adam_dense_67_bias_m:/<
*assignvariableop_29_adam_dense_68_kernel_m://6
(assignvariableop_30_adam_dense_68_bias_m:/<
*assignvariableop_31_adam_dense_69_kernel_m://6
(assignvariableop_32_adam_dense_69_bias_m:/<
*assignvariableop_33_adam_dense_70_kernel_m://6
(assignvariableop_34_adam_dense_70_bias_m:/<
*assignvariableop_35_adam_dense_71_kernel_m://6
(assignvariableop_36_adam_dense_71_bias_m:/<
*assignvariableop_37_adam_dense_72_kernel_m:/6
(assignvariableop_38_adam_dense_72_bias_m:<
*assignvariableop_39_adam_dense_65_kernel_v:6
(assignvariableop_40_adam_dense_65_bias_v:<
*assignvariableop_41_adam_dense_66_kernel_v:/6
(assignvariableop_42_adam_dense_66_bias_v:/<
*assignvariableop_43_adam_dense_67_kernel_v://6
(assignvariableop_44_adam_dense_67_bias_v:/<
*assignvariableop_45_adam_dense_68_kernel_v://6
(assignvariableop_46_adam_dense_68_bias_v:/<
*assignvariableop_47_adam_dense_69_kernel_v://6
(assignvariableop_48_adam_dense_69_bias_v:/<
*assignvariableop_49_adam_dense_70_kernel_v://6
(assignvariableop_50_adam_dense_70_bias_v:/<
*assignvariableop_51_adam_dense_71_kernel_v://6
(assignvariableop_52_adam_dense_71_bias_v:/<
*assignvariableop_53_adam_dense_72_kernel_v:/6
(assignvariableop_54_adam_dense_72_bias_v:
identity_56¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9È
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*Ô
valueÊBÇ8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÿ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÆ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_65_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_65_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_66_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_66_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_67_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_67_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_68_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_68_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_69_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_69_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_70_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_70_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_71_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_71_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_72_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_72_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¥
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17§
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18§
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¦
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¡
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23²
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_65_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_65_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_66_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_66_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_67_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_67_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_68_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_68_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_69_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_69_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_70_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_70_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_71_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_71_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_72_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_72_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_65_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_65_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41²
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_66_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42°
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_66_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_67_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_67_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45²
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_68_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46°
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_68_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47²
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_69_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48°
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_69_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49²
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_70_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50°
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_70_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51²
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_71_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52°
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_71_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53²
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_72_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54°
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_72_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


/__inference_sequential_9_layer_call_fn_90772363

inputs
unknown:
	unknown_0:
	unknown_1:/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5://
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_907719232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

÷
F__inference_dense_70_layer_call_and_return_conditional_losses_90772638

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¬

+__inference_dense_71_layer_call_fn_90772647

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_907719002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_65_layer_call_and_return_conditional_losses_90771798

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£-
±
J__inference_sequential_9_layer_call_and_return_conditional_losses_90771923

inputs#
dense_65_90771799:
dense_65_90771801:#
dense_66_90771816:/
dense_66_90771818:/#
dense_67_90771833://
dense_67_90771835:/#
dense_68_90771850://
dense_68_90771852:/#
dense_69_90771867://
dense_69_90771869:/#
dense_70_90771884://
dense_70_90771886:/#
dense_71_90771901://
dense_71_90771903:/#
dense_72_90771917:/
dense_72_90771919:
identity¢ dense_65/StatefulPartitionedCall¢ dense_66/StatefulPartitionedCall¢ dense_67/StatefulPartitionedCall¢ dense_68/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall§
 dense_65/StatefulPartitionedCallStatefulPartitionedCallinputsdense_65_90771799dense_65_90771801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_907717982"
 dense_65/StatefulPartitionedCallÊ
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_90771816dense_66_90771818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_907718152"
 dense_66/StatefulPartitionedCallÊ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_90771833dense_67_90771835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_907718322"
 dense_67/StatefulPartitionedCallÊ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_90771850dense_68_90771852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_907718492"
 dense_68/StatefulPartitionedCallÊ
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_90771867dense_69_90771869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_907718662"
 dense_69/StatefulPartitionedCallÊ
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_90771884dense_70_90771886*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_907718832"
 dense_70/StatefulPartitionedCallÊ
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_90771901dense_71_90771903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_907719002"
 dense_71/StatefulPartitionedCallÊ
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_90771917dense_72_90771919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_72_layer_call_and_return_conditional_losses_907719162"
 dense_72/StatefulPartitionedCall
IdentityIdentity)dense_72/StatefulPartitionedCall:output:0!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


/__inference_sequential_9_layer_call_fn_90772400

inputs
unknown:
	unknown_0:
	unknown_1:/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5://
	unknown_6:/
	unknown_7://
	unknown_8:/
	unknown_9://

unknown_10:/

unknown_11://

unknown_12:/

unknown_13:/

unknown_14:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_907721212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»-
¹
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772237
dense_65_input#
dense_65_90772196:
dense_65_90772198:#
dense_66_90772201:/
dense_66_90772203:/#
dense_67_90772206://
dense_67_90772208:/#
dense_68_90772211://
dense_68_90772213:/#
dense_69_90772216://
dense_69_90772218:/#
dense_70_90772221://
dense_70_90772223:/#
dense_71_90772226://
dense_71_90772228:/#
dense_72_90772231:/
dense_72_90772233:
identity¢ dense_65/StatefulPartitionedCall¢ dense_66/StatefulPartitionedCall¢ dense_67/StatefulPartitionedCall¢ dense_68/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall¯
 dense_65/StatefulPartitionedCallStatefulPartitionedCalldense_65_inputdense_65_90772196dense_65_90772198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_907717982"
 dense_65/StatefulPartitionedCallÊ
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_90772201dense_66_90772203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_907718152"
 dense_66/StatefulPartitionedCallÊ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_90772206dense_67_90772208*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_907718322"
 dense_67/StatefulPartitionedCallÊ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_90772211dense_68_90772213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_907718492"
 dense_68/StatefulPartitionedCallÊ
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_90772216dense_69_90772218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_907718662"
 dense_69/StatefulPartitionedCallÊ
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_90772221dense_70_90772223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_907718832"
 dense_70/StatefulPartitionedCallÊ
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_90772226dense_71_90772228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_907719002"
 dense_71/StatefulPartitionedCallÊ
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_90772231dense_72_90772233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU 

XLA_CPU2J 8 *O
fJRH
F__inference_dense_72_layer_call_and_return_conditional_losses_907719162"
 dense_72/StatefulPartitionedCall
IdentityIdentity)dense_72/StatefulPartitionedCall:output:0!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_65_input
Ò	
÷
F__inference_dense_72_layer_call_and_return_conditional_losses_90772677

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_67_layer_call_and_return_conditional_losses_90771832

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_66_layer_call_and_return_conditional_losses_90771815

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

÷
F__inference_dense_69_layer_call_and_return_conditional_losses_90772618

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_71_layer_call_and_return_conditional_losses_90771900

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
®

÷
F__inference_dense_71_layer_call_and_return_conditional_losses_90772658

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
ìK

J__inference_sequential_9_layer_call_and_return_conditional_losses_90772459

inputs9
'dense_65_matmul_readvariableop_resource:6
(dense_65_biasadd_readvariableop_resource:9
'dense_66_matmul_readvariableop_resource:/6
(dense_66_biasadd_readvariableop_resource:/9
'dense_67_matmul_readvariableop_resource://6
(dense_67_biasadd_readvariableop_resource:/9
'dense_68_matmul_readvariableop_resource://6
(dense_68_biasadd_readvariableop_resource:/9
'dense_69_matmul_readvariableop_resource://6
(dense_69_biasadd_readvariableop_resource:/9
'dense_70_matmul_readvariableop_resource://6
(dense_70_biasadd_readvariableop_resource:/9
'dense_71_matmul_readvariableop_resource://6
(dense_71_biasadd_readvariableop_resource:/9
'dense_72_matmul_readvariableop_resource:/6
(dense_72_biasadd_readvariableop_resource:
identity¢dense_65/BiasAdd/ReadVariableOp¢dense_65/MatMul/ReadVariableOp¢dense_66/BiasAdd/ReadVariableOp¢dense_66/MatMul/ReadVariableOp¢dense_67/BiasAdd/ReadVariableOp¢dense_67/MatMul/ReadVariableOp¢dense_68/BiasAdd/ReadVariableOp¢dense_68/MatMul/ReadVariableOp¢dense_69/BiasAdd/ReadVariableOp¢dense_69/MatMul/ReadVariableOp¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢dense_72/BiasAdd/ReadVariableOp¢dense_72/MatMul/ReadVariableOp¨
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_65/MatMul/ReadVariableOp
dense_65/MatMulMatMulinputs&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_65/MatMul§
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_65/BiasAdd/ReadVariableOp¥
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_65/BiasAdds
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_65/Relu¨
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02 
dense_66/MatMul/ReadVariableOp£
dense_66/MatMulMatMuldense_65/Relu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_66/MatMul§
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_66/BiasAdd/ReadVariableOp¥
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_66/BiasAdds
dense_66/ReluReludense_66/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_66/Relu¨
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_67/MatMul/ReadVariableOp£
dense_67/MatMulMatMuldense_66/Relu:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_67/MatMul§
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_67/BiasAdd/ReadVariableOp¥
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_67/BiasAdds
dense_67/ReluReludense_67/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_67/Relu¨
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_68/MatMul/ReadVariableOp£
dense_68/MatMulMatMuldense_67/Relu:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_68/MatMul§
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_68/BiasAdd/ReadVariableOp¥
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_68/BiasAdds
dense_68/ReluReludense_68/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_68/Relu¨
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_69/MatMul/ReadVariableOp£
dense_69/MatMulMatMuldense_68/Relu:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_69/MatMul§
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_69/BiasAdd/ReadVariableOp¥
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_69/BiasAdds
dense_69/ReluReludense_69/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_69/Relu¨
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_70/MatMul/ReadVariableOp£
dense_70/MatMulMatMuldense_69/Relu:activations:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_70/MatMul§
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_70/BiasAdd/ReadVariableOp¥
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_70/BiasAdds
dense_70/ReluReludense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_70/Relu¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

://*
dtype02 
dense_71/MatMul/ReadVariableOp£
dense_71/MatMulMatMuldense_70/Relu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_71/BiasAdds
dense_71/ReluReludense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/2
dense_71/Relu¨
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:/*
dtype02 
dense_72/MatMul/ReadVariableOp£
dense_72/MatMulMatMuldense_71/Relu:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/MatMul§
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_72/BiasAdd/ReadVariableOp¥
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/BiasAdd
IdentityIdentitydense_72/BiasAdd:output:0 ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

÷
F__inference_dense_65_layer_call_and_return_conditional_losses_90772538

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¹
serving_default¥
I
dense_65_input7
 serving_default_dense_65_input:0ÿÿÿÿÿÿÿÿÿ<
dense_720
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ç
©K
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
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"£G
_tf_keras_sequentialG{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_65_input"}}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_69", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 7]}, "float32", "dense_65_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_65_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_69", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}, {"class_name": "Dense", "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
¿	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerþ{"name": "dense_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7]}, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
Ï

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"name": "dense_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_66", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
Ñ

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "dense_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_67", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 47}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47]}}
Ô

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
__call__
+ &call_and_return_all_conditional_losses"­
_tf_keras_layer{"name": "dense_68", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_68", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 47}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47]}}
Ô

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"­
_tf_keras_layer{"name": "dense_69", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_69", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 47}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47]}}
Ô

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"­
_tf_keras_layer{"name": "dense_70", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 47}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47]}}
Ô

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"­
_tf_keras_layer{"name": "dense_71", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 47, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 47}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47]}}
Õ

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"®
_tf_keras_layer{"name": "dense_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 47}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47]}}

?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemvmwmxmymzm{!m|"m}'m~(m-m.m3m4m9m:mvvvvvv!v"v'v(v-v.v3v4v9v:v"
	optimizer
 "
trackable_list_wrapper

0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper

0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper
Î

regularization_losses
Dlayer_metrics
Enon_trainable_variables

Flayers
Gmetrics
	variables
Hlayer_regularization_losses
trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
©serving_default"
signature_map
!:2dense_65/kernel
:2dense_65/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Ilayer_metrics
Jnon_trainable_variables

Klayers
Lmetrics
	variables
Mlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:/2dense_66/kernel
:/2dense_66/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Nlayer_metrics
Onon_trainable_variables

Players
Qmetrics
	variables
Rlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!://2dense_67/kernel
:/2dense_67/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Slayer_metrics
Tnon_trainable_variables

Ulayers
Vmetrics
	variables
Wlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!://2dense_68/kernel
:/2dense_68/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
°
#regularization_losses
Xlayer_metrics
Ynon_trainable_variables

Zlayers
[metrics
$	variables
\layer_regularization_losses
%trainable_variables
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
!://2dense_69/kernel
:/2dense_69/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
°
)regularization_losses
]layer_metrics
^non_trainable_variables

_layers
`metrics
*	variables
alayer_regularization_losses
+trainable_variables
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
!://2dense_70/kernel
:/2dense_70/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
°
/regularization_losses
blayer_metrics
cnon_trainable_variables

dlayers
emetrics
0	variables
flayer_regularization_losses
1trainable_variables
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
!://2dense_71/kernel
:/2dense_71/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
°
5regularization_losses
glayer_metrics
hnon_trainable_variables

ilayers
jmetrics
6	variables
klayer_regularization_losses
7trainable_variables
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
!:/2dense_72/kernel
:2dense_72/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
°
;regularization_losses
llayer_metrics
mnon_trainable_variables

nlayers
ometrics
<	variables
player_regularization_losses
=trainable_variables
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
q0"
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
Ô
	rtotal
	scount
t	variables
u	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 34}
:  (2total
:  (2count
.
r0
s1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
&:$2Adam/dense_65/kernel/m
 :2Adam/dense_65/bias/m
&:$/2Adam/dense_66/kernel/m
 :/2Adam/dense_66/bias/m
&:$//2Adam/dense_67/kernel/m
 :/2Adam/dense_67/bias/m
&:$//2Adam/dense_68/kernel/m
 :/2Adam/dense_68/bias/m
&:$//2Adam/dense_69/kernel/m
 :/2Adam/dense_69/bias/m
&:$//2Adam/dense_70/kernel/m
 :/2Adam/dense_70/bias/m
&:$//2Adam/dense_71/kernel/m
 :/2Adam/dense_71/bias/m
&:$/2Adam/dense_72/kernel/m
 :2Adam/dense_72/bias/m
&:$2Adam/dense_65/kernel/v
 :2Adam/dense_65/bias/v
&:$/2Adam/dense_66/kernel/v
 :/2Adam/dense_66/bias/v
&:$//2Adam/dense_67/kernel/v
 :/2Adam/dense_67/bias/v
&:$//2Adam/dense_68/kernel/v
 :/2Adam/dense_68/bias/v
&:$//2Adam/dense_69/kernel/v
 :/2Adam/dense_69/bias/v
&:$//2Adam/dense_70/kernel/v
 :/2Adam/dense_70/bias/v
&:$//2Adam/dense_71/kernel/v
 :/2Adam/dense_71/bias/v
&:$/2Adam/dense_72/kernel/v
 :2Adam/dense_72/bias/v
2
/__inference_sequential_9_layer_call_fn_90771958
/__inference_sequential_9_layer_call_fn_90772363
/__inference_sequential_9_layer_call_fn_90772400
/__inference_sequential_9_layer_call_fn_90772193À
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
ö2ó
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772459
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772518
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772237
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772281À
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
è2å
#__inference__wrapped_model_90771780½
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
dense_65_inputÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_dense_65_layer_call_fn_90772527¢
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
F__inference_dense_65_layer_call_and_return_conditional_losses_90772538¢
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
+__inference_dense_66_layer_call_fn_90772547¢
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
F__inference_dense_66_layer_call_and_return_conditional_losses_90772558¢
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
+__inference_dense_67_layer_call_fn_90772567¢
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
F__inference_dense_67_layer_call_and_return_conditional_losses_90772578¢
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
+__inference_dense_68_layer_call_fn_90772587¢
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
F__inference_dense_68_layer_call_and_return_conditional_losses_90772598¢
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
+__inference_dense_69_layer_call_fn_90772607¢
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
F__inference_dense_69_layer_call_and_return_conditional_losses_90772618¢
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
+__inference_dense_70_layer_call_fn_90772627¢
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
F__inference_dense_70_layer_call_and_return_conditional_losses_90772638¢
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
+__inference_dense_71_layer_call_fn_90772647¢
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
F__inference_dense_71_layer_call_and_return_conditional_losses_90772658¢
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
+__inference_dense_72_layer_call_fn_90772667¢
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
F__inference_dense_72_layer_call_and_return_conditional_losses_90772677¢
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
&__inference_signature_wrapper_90772326dense_65_input"
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
 ¨
#__inference__wrapped_model_90771780!"'(-.349:7¢4
-¢*
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_72"
dense_72ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_65_layer_call_and_return_conditional_losses_90772538\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_65_layer_call_fn_90772527O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_66_layer_call_and_return_conditional_losses_90772558\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ~
+__inference_dense_66_layer_call_fn_90772547O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ/¦
F__inference_dense_67_layer_call_and_return_conditional_losses_90772578\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ~
+__inference_dense_67_layer_call_fn_90772567O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¦
F__inference_dense_68_layer_call_and_return_conditional_losses_90772598\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ~
+__inference_dense_68_layer_call_fn_90772587O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¦
F__inference_dense_69_layer_call_and_return_conditional_losses_90772618\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ~
+__inference_dense_69_layer_call_fn_90772607O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¦
F__inference_dense_70_layer_call_and_return_conditional_losses_90772638\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ~
+__inference_dense_70_layer_call_fn_90772627O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¦
F__inference_dense_71_layer_call_and_return_conditional_losses_90772658\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 ~
+__inference_dense_71_layer_call_fn_90772647O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¦
F__inference_dense_72_layer_call_and_return_conditional_losses_90772677\9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_72_layer_call_fn_90772667O9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿÈ
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772237z!"'(-.349:?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772281z!"'(-.349:?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772459r!"'(-.349:7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_9_layer_call_and_return_conditional_losses_90772518r!"'(-.349:7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
/__inference_sequential_9_layer_call_fn_90771958m!"'(-.349:?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
/__inference_sequential_9_layer_call_fn_90772193m!"'(-.349:?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_9_layer_call_fn_90772363e!"'(-.349:7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_9_layer_call_fn_90772400e!"'(-.349:7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ½
&__inference_signature_wrapper_90772326!"'(-.349:I¢F
¢ 
?ª<
:
dense_65_input(%
dense_65_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_72"
dense_72ÿÿÿÿÿÿÿÿÿ