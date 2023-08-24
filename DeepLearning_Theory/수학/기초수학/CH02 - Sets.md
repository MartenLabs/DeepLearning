
### Definition and Notations of Sets
- distinct
- doesn't change from person to person
- 서로 같은 종류의 object들


### Notations
- Rotster Form
	- $Set = \lbrace element_1, element_2, ..., element_n\rbrace$
	 - A = {1, 2, 3, 4}
	 - C = {red, green, blue}

- Set Builder 
	- Set  = {element | element's condition}
	- $A = \lbrace x | 1 \leq x \in N \leq 4\rbrace$
	- C = {x | x는 빛의 삼원색}


### Univertal/Empty Sets
- Empty Sets
	- $\emptyset = \lbrace \rbrace$
- Universal Sets
	- U = 가능한 모든 원소들의 집합


### Common Number Sets
- Natural Numbers(자연수)
	- $N=\lbrace 1, 2, 3, ...\rbrace$
	- N = { x | (x는 자연수)}

- Whole Number
	- $W = \lbrace 0, 1, 2, ...\rbrace$
	- $W = \lbrace x\; |\; (x는\; 0 \vee x는\; 자연수)\rbrace$

- Integers(정수)
	- $Z = \lbrace ..., -2, -1, 0, 1, 2, ...\rbrace$
	- $Z = \lbrace x\; |\; (x는\; 정수)\rbrace$

- Rational Numbers(유리수)
	- $Q = \lbrace {1\over2}, {1\over3}, {3\over2}, {100\over101}, ...\rbrace$
	- $\lbrace x\; |\; x = {p\over q}, (p,q\;  are\; integers\, \vee\, q \neq0\rbrace$

- Irrational Numbers(무리수)
	- $I = \lbrace \pi, e, \sqrt 2, ...\rbrace$
	- $I = \lbrace x\; | \neg (x는\, 유리수)\rbrace$
	- $\neg$ : logic negation

- Real Numbers(실수)
	- $R = \lbrace x\, |\, (x는 \, 유리수\, \vee\, x는\, 무리수) \rbrace$

- Complex Numbers(복소수)
	- $C = \lbrace a + j\cdot b \, |\, (a, b는 실수)\rbrace$


### Coordinate Spaces
- Coordinate Plane : $R^2 = \lbrace (x, y) | x, y 는\, 실수\rbrace$               (좌표평면)
- Coordinate Space : $R^3 = \lbrace (x, y, z)\, |\, x, y, z는\, 실수\rbrace$    (좌표공간)
- Higher Dimentional Spaces 
	- $R^n = \lbrace (x_1, x_2, ..., x_n)\, | \, \forall x_i 는\, 실수  \rbrace$


### Intersections 
- No intersection      => $S = \emptyset$                            (교점 x)
- Point intersection => $S = \lbrace (x_s, y_s, z_s)\rbrace$       (교점 1개)
- Lines intersection => $S = \lbrace r\, |\, (r\; is\; on\, L)\rbrace$ (무수히 많은 교점)


### Cardinality of Sets

$\vert A \vert$ = (# elements) - 집합의 크기 = 원소의 갯수


###### - Cardinality of Empty Set 
- $\vert \emptyset \vert = 0$

###### - Singleton Set
- $\vert A \vert = 1$

###### - Equivalent Sets
- $\vert A \vert = \vert B \vert$



### Finite / Infinite Sets

- ##### Finite Sets (유한집합)

- ##### Encoding of Elements
	- 주로 컴퓨터의 연산을 위해, 원소들을 index에 대응시키는 과정

- ##### Infinite Sets
	- ex) N, W, Z, Q, I, R, C

- ##### Countably Infinite Sets
	- 원소들을 index에 대응시킬 수 있는 무한집합
	- encoding이 가능함 
	- ex) N, W, Z, Q

- Uncountably Infinite Sets
	- 원소들을 index에 대응시킬 수 없는 무한집합 
	- ex) R, C



### Equal Sets
$A = B \leftrightarrow [(\forall a \in A) \in B]\; \wedge \; [(\forall b \in B)\; \in\; A]$
- $\forall$ 모든 원소



### Inclusion / Exclusion of Sets

- ##### Subset 
	- 집합 A의 모든 원소가 집합 B에 포함될 때, A는 B의 subset이라 한다
	- $A \subseteq B \leftrightarrow (\forall a \in A) \in B$

- ##### Superset
	- 집합 B의 모든 원소가 A에 포함될 때, A는 B의 superset이라 한다
	- $A \supseteq B \leftrightarrow (\forall b \in B) \in A$



### Inclusion/Exclusion of Sets

- ex) A = {a, b, c, d}
	- $\emptyset \subseteq A$
	- $\lbrace a, b, c, d \rbrace \subseteq A \leftrightarrow A\subseteq A$ : A is a subset of itself.
	- $\lbrace a, b, c, d \rbrace \supseteq A \leftrightarrow A \supseteq A$
	- A는 A의 subset이자 superset



### Proper Subsets / Supersets

- ##### Proper Subsets
	- 집합 A, B에 대해  <u>A가 B의 subse</u>t 이지만 <u>완전히 같지는 않을 때</u>,     A는 B의 proper subset이라 한다. 
	-  적어도 <u>B의 원소 중 하나</u>는 <u>A에 포함되지 않아야 한다</u>
$$A \subset B \leftrightarrow [(\forall a \in A) \in B] \wedge [A \neq B]$$
		- ex) 
			$\emptyset \subset A$
			A is not a proper subset of A

- ##### Proper Supersets
	- <u>A가 B의 superset이지만 완전히 같지는 않을때</u>
	- 적어도 A의 원소 중 하나는 B에 포함되지 않아야 한다 
$$A \supset B \leftrightarrow [(\forall b \in B) \in A] \wedge [A \neq B]$$
		- ex)
			A is not a proper superset of A



### Unary / Binary Operations

##### Operations on Sets 
- 일정한 규칙을 통해 새로운 집합을 만들어내는 과정 

##### Unary Operations $f: A\rightarrow B$
- 집합 하나로 연산 ex) $\sqrt4 = 2$
- power set of sets 
- complement of sets 

##### Binary Operations $f: A \times B \rightarrow C$
- 집합 2개로 새로운 집합 생성 ex) 3 + 2 = 5
- union of sets
- set difference
- symmetric difference
- Cartesian product of sets 



### Unary Operations - Power Sets 
- ##### Power Sets
	- 집합 A의 모든 subset들의 집합 P(A)   $P(A) = \lbrace X | X \subseteq A\rbrace$
	- 모든 원소들은 집합

- Power Set and Cardinality 
$$\vert P(A)\vert = 2^{\vert A\vert}$$
- ex)
	- A = {0, 1} 일 때

	- Subsets => $\emptyset$, {0}, {1}, {0, 1}
	- $\emptyset \in$ P(A)
	- {0} $\in$ P(A)
	- {1} $\in$ P(A)
	- {0, 1} $\in$ P(A)
	- P(A) = {$\emptyset$, {0}, {1}, {0, 1}}


#### <u>Note!</u> 
1. Power set의 원소들은 집합
2. $\emptyset$ 과 A는 P(A)의 원소 
3. P(A)의 원소들은 binary number로 인코딩 가능



### Unary Operations - Complements 
- A에 포함되지 않은 원소들을 모은 집합을 A의 complement 라 함,  $A^c$ 
- $A^c = \lbrace x | x \notin A \rbrace$

- ##### Cardinality 
	- $\vert A^c \vert = \vert U \vert - \vert A \vert$



### Binary Operations - Intersections and Unions
- 집합 A, B에 모두 포함되는 원소들의 모든 집합을 A와 B의 intersection이라 부름, $A \cap B$
	- $C = A \cap B = \lbrace x \vert (x \in A) \wedge (x \in B)\rbrace$ == $AB$

- 집합 A 또는 B에 포함되는 원소들의 모든 집합을 A와 B의 union 이라 부름, $A \cup B$
	- $C = A\cup B = \lbrace x | (x \in A) \vee (x \in B)\rbrace$



### Binary Operations - Intersections and Unions
- ##### Cardinality
	- $\vert A \cup B \vert = \vert A\vert + \vert B \vert - \vert A \cap B \vert$



### Binary Operations - Interections and Unions
- General Intersections
	- $\cap_{i = 1}^{n} A_i = A_1 \cap A_2 \cap ... \cap A_n$   or    $\cap_{i = 1}^{n} A_i = A_1A_2...A_n$



### Binary Operations - Intersections and Unions
- General Unions
	- $\cup_{i=1}^n A_i = A_1 \cup A_2 \cup ... \cup A_n$ 



### Binary Operations  - Intersections and Unions

- ##### The Algebraic Propertions
	- Commutative Law
		- $A \cup B = B \cup A$
		- $A \cap B = B \cap A$
	
	 - Associative Law
		 - $(A \cup B) \cup C = A \cup (B \cup C)$
		 - $(A \cap B) \cap C = A \cap (B \cap C)$

	- Distributive Law
		- $A \cap (B\cup C) = (A\cap B) \cup (A\cap C)$


- ##### Identities
	1. $A \cup \emptyset = A$
	2. $A \cap \emptyset = \emptyset$
	3. $A \cup U = U$
	4. $A \cap U = A$
	5. $A \cup A^c = U$
	6. $A \cap A^c = \emptyset$
	7. $(A^c)^c = A$


- ##### De Morgan's Law
	- $(A \cap B)^c = A^c \cup B^c$
	- $(A \cup B)^c = A^c \cap B^c$




### Binary Operations - Set Differences 

- 집합 A, B에 대해 A에는 포함되고, B에는 포함되지 않는 원소들을 모은 집합을 A - B 로 나타내고 set difference (차집합) 이라 부름 
$$A - B = \lbrace x | (x \in A) \wedge (x \notin B)\rbrace$$
- A - B 는 A \ B로 표현하기도 함
- set difference 는 relative complement 라 부르기도 한다


- ##### Computaion Exercises
	1. A - B = $A \cap B^c$
	2. B - A = $B \cap A^c$
	3. A - B = $A - (A \cap B) \;= \; (A \cup B) - B$
	4. B - A = $B - (A \cap B) \; = \; (A \cup B) - A$
	5. $U - A = U \cap A^c$


- ##### The Algebraic Properties
	- Anti-commutativity $A - B \neq B - A$
	- Anti-associativity    $A - (B - C) \neq (A - B) - C$

	- Distributive Law
		1.  $C - (A \cap B) = C \cap (A \cap B)^c$
					= $C \cap (A^c \cup B^c)$
					= $(C \cap A^c) \cup (C \cap B^c)$
					= $(C - A) \cup (C - B)$

		2. $C - (A \cup B) = C \cap (A\cup B)^c$
					= $C \cap (A^c \cap B^c)$
					= $(C \cap A^c) \cap (C \cap B^c)$
					= $(C - A) \cap (C - B)$
		
		$\cup \rightarrow \cap,\; \cap \rightarrow \cup$




### Binary Operations - Symmetric Differences
- 집합 A, B에 대해 A - B와 B - A의 union
$$
\begin{aligned}
A \triangle B = \lbrace x \vert \; (A-B) \cup (B-A) \rbrace \\ 
= \lbrace x \vert \; [(x \in A) \vee (x \in B)] \wedge (x \notin A \cap B)  \rbrace
\end{aligned}
$$



### Binary Operations - Cartesian Product
-  집합 A, B에서 원소 a, b 들을 각각 뽑아 (a, b)를 만들 때, 모든 (a, b)들의 집합을 A x B라 한다.
$$A \times B = \lbrace (a, b)\; \vert \; (a \in A) \wedge (b \in B)\rbrace  $$

- $A = \lbrace a_1, a_2, ..., a_m \rbrace, \;\; \vert A \vert = m$ 
- $B = \lbrace b_1, b_2, ..., b_n \rbrace, \;\;\;\; \vert B \vert = n$ 

|          |    $b_1$     |    $b_2$     | $\dots$ |    $b_n$     |
|:--------:|:------------:|:------------:|:-------:|:------------:|
|  $a_1$   | $(a_1, b_1)$ | $(a_1, b_2)$ | $\dots$ | $(a_1, b_n)$ |
|  $a_2$   | $(a_2, b_1)$ | $(a_2, b_2)$ | $\dots$ | $(a_2, b_n)$ |
| $\vdots$ |   $\vdots$   |   $\vdots$   | $\dots$ |   $\vdots$   |
|  $a_m$   | $(a_m, b_1)$ | $(a_m, b_2)$ | $\dots$ | $(a_m, b_n)$             |

- A x B = {$(a_1, b_1), (a_1, b_2), ... , (a_1, b_n)$
		 $(a_2, b_1), (a_2, b_2), \dots, (a_2, b_n)$
		 $...$
		 $(a_m, b_1), (a_m, b_2), \dots, (a_m, b_n) \rbrace$

- $\vert A \times B \vert = m \times n$




### Disjoint Sets
-  집합 A, B에 대해 $A \cap B = \emptyset$일 때, A, B는 disjoint set이다.
- disjoint는 mutually exclusive라고도 부른다.

- ##### Cardinality 
	- $\vert A \cap B \vert = 0$
	- $\vert A \cup B \vert = \vert A \vert + \vert B \vert$




### Partitions of Sets
- Universal set($U$)안에 n개의 집합 $A_1, A_2, ..., A_n$ 이 있고, $U의 모든 원소들이 모두 단 하나의 $A_i$에만 포함될 때, $\lbrace A_1, A_2, ... , A_n \rbrace$ 를 $U$의 partition이라 부른다. 

$$A_i \cap A_j = \emptyset, \; i \neq j$$
$$\cup_{i = 1}^n A_i = U$$

- ##### Non-uniqueness of Partitions
	- Partitions of Sets is NOT UNIQUE (나누는 방법은 여러가지)


- ##### Making a Complete Set with Partitions
$$ B = \cup_{i = 1}^n (A_i \cap B)$$
