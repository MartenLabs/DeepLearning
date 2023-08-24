
### Propositions (명제)
- True, False 로 판단할 수 있는 문장
- 참일 수도, 거짓일 수도 있는 경우 확률의 개념이 도입된다.


### Axioms (공리)
- 참으로 증명없이 받아들이는 proposition


### Conditions 
- 변수에 따라 Ture, False가 달라지는 식 
- 항상 Ture이거나 False이면 각각 true proposition, false proposition으로 생각한다.
- ex) $x^2 = 4 \rightarrow True$, if $x = 2$ or $x = -2$
		      $\rightarrow False$, if otherwise
		      

### Truth Sets
- 어떤 condition을 만족하는 원소들의 집합 
- condition이 정해지면 truth set이 생김 





### Unary / Binary Operations

- ##### Unary Operations
	- Logical Negation($\neg$) - not

- ##### Binary Operations
	- Logical Conjunction ($\wedge$)            - and 
	- Logical Disjunction ($\vee$)              - or
	- Logical Exclusive Disjunction ($\oplus$)
	- Logical Implication ($\rightarrow$)             - if then
	- NAND, NOR, XNOR



### Truth Tables
- input condition(s)이 true / false인지에 따라 output condition의    true / false 를 정리한 표




### Logical Negation($\neg$) - NOT 연산
- input condition의 참과 거짓을 바꾸는 연산 


### Logical Conjunction ($\wedge$) - AND 연산
- 두 input condition A, B가 모두 참일 때만 output condition이 참인 연산 


### Logical Disjunction ($\vee$) - OR 연산
- 두 input condition A, B 중 하나라도 참이면 output condition이 참                 


### Logical Exclusive Disjunction ($\oplus$) - XOR 연산
- 두 input condition A, B 중 하나만 참일때, output condition이 참인 연산
- $\oplus$ 는 입력이 서로 같은지, 다른지를 판별하는 연산으로도 사용됨




### Logical Implications
- 조건 A가 만족될 때, B가 만족됨을 추론하는 연산
	- If A, then B
$$A \rightarrow B$$
			A = promise (약속 - input)
	        B = conclusion (결과 - output)

- ##### Innocent Until Proven Guilty
	- A is True, but B is False => 유죄가 입증되기 전까진 무죄

	- ex) 100점을 맞으면 100만원을 주기로 함 
	- $A \rightarrow B$
Truth Table of Logical Implication
| A(100점 맞음) | B(100만원 줌)  | Y (유/무죄) |
| :-------------: | :--------------: | :-----------: |
| 0             | 0              | 1(무죄)     |
| 0(노력했지만 못맞음)             | 1(기특해서 줌) | 1(무죄)     |
| 1             | 0(안줌)        | 0(유죄)     |
| 1             | 1              | 1            |




### Converse, Inverse, Contrapositive

$$ A \rightarrow B$$
- ##### Converse (역)       $B \rightarrow A$
- ##### Inverse     (이)        $\neg A \rightarrow \neg B$
- ##### Contrapositive (대우)   $\neg B \rightarrow \neg A$




### Sufficiency and Necessity 

- condition p, q와 이 조건들에 대한 각각의 truth set P, Q일 때,                        $p \rightarrow q$ 가 참이기 위해선 $P \subseteq Q$ 여야 한다


$$p \rightarrow q$$
- ##### Sufficiency (충분조건)
	- p는 q이기 위한 충분조건이다.
	- q가 참이기 위해 p가 참임 만으로 충분하다
	- ex) $x$가 사람이면 자동으로 동물을 만족

- ##### Necessity (필요조건)
	- q는 p이기 위한 필요조건이다.
	- p가 참이기 위해 <u>q가 참임</u> 이 필요하다



- ##### if Condition
	- if and only if
		- 두 조건 $p$, $q$ 가 서로 같은 truth set을 만들 때,                               두 조건은 같은 조건이다.
		- ex) $p: x^2 = 4 \leftrightarrow q : x = \pm 2$ 


#####  $\forall$   For All
- $\forall x,\, p$ 
- $x$가 될 수 있는 모든 경우에 대해 조건 $p$가 만족된다. 
- ex) $\forall$ 강아지는 동물이다

##### $\exists$   There Exists
- $\exists x, \; q$ 
- 어떤(최소한 하나 이상의) x의 값이 조건 q를 만족시킨다.
- ex) $\exists x, \, x + 3 = 0$   



### Identities and Inverses 
- Additive Identities and Inverses
	- $\forall x, \; \exists a, \; x + a = x \rightarrow a = 0$ (항등원)
	- $\forall x, \exists a, x + a = 0 \rightarrow a = -x$ (역원)

- Multiplicative Identities and Inverses
	- $\forall x$ except $x = 0, \; \exists a,$ $x \cdot a = x \rightarrow a = 1$ (항등원)
	- $\forall x$ except $x = 0,$ $\exists a,$ $a \cdot a = 1 \rightarrow a = x ^{-x}$ (역원)



### Logical Equivalence
- $(A \rightarrow B) \leftrightarrow (\neg B \rightarrow \neg A)$
- $(A \oplus B) \leftrightarrow [(A \wedge (\neg B)) \vee ((\neg A) \wedge B)]$


### De Morgan's Law
- $\neg (A \wedge B) \leftrightarrow (\neg A) \vee (\neg B)$
- $\neg (A \vee B) \leftrightarrow (\neg A) \wedge (\neg B)$


### Other Equivalences 
1. $\neg (\neg A) \leftrightarrow A$ 
2. $A \wedge A \leftrightarrow A$
3. $A \vee A \leftrightarrow A$

- ##### Commutativity
	1. $A \wedge B \leftrightarrow B \wedge A$
	2. $A \vee B \leftrightarrow B \vee A$

- Associativity
	1. $(A \wedge B) \wedge C \leftrightarrow A \wedge (B \wedge C)$
	2. $(A \vee B) \vee C \leftrightarrow A \vee (B \vee C)$

- Distributivity
	1. $A \wedge (B \vee C) \leftrightarrow (A \wedge B) \vee (A \wedge C)$
	2. $A \vee (B \wedge C) \leftrightarrow (A \vee B) \wedge (A \vee C)$




### Tautologies and Contradictions
- ##### Tautologies
	- 항상 결과가 참이 되는 연산 
	- ex) 
		- $A \vee (\neg A)$
		- $[(A \rightarrow B) \wedge (B \rightarrow C)] \rightarrow (A \rightarrow C)$

- ##### Contradictions
	- 항상 결과가 거짓이 되는 연산 
	- ex)
		- $A \wedge (\neg A)$
		- $(A \vee B) \wedge [(\neg A) \wedge (\neg B)]$


