MATH_LEVEL2= '''Question: What is the largest value of $x$ such that the expression \[\dfrac{x+1}{8x^2-65x+8}\] is not defined?
Let's think step by step
In this particular case, the fraction will be undefined only if its denominator is equal to zero. 
Because of this, we can ignore the numerator. 
We start by setting the binomial in the denominator equal to 0: 
\\begin{align*} 8x^2-65x+8=0
\\\Rightarrow\qquad (8x-1)(x-8)=0
\end{align*} 
We find that the two possible values for $x$ are $\\frac18$ and $8$. 
Since the question asks for the largest value, the final solution is $\\boxed{8}$.
The answer is 8

Question: BoatWorks built 3 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatWorks by the end of March of this year?
Let's think step by step
The number of boats built is $3+3\cdot2+3\cdot2^2 = 3+6+12 = \\boxed{21}$.
The answer is 21

Question: If $5a+2b=0$ and $a$ is two less than $b$, what is $7b$?
Let's think step by step
First we begin by solving the system of equations \\begin{align*}
5a+2b&=0, \\\\
b-2&=a.
\end{align*}
Making the substitution for $a$ from the second equation to the first, we get $5(b-2)+2b=0$, 
which simplifies to $7b-10=0$. 
Solving for $b$, we find that $b=\\frac{10}{7}$. 
Hence $7b=7\cdot \\frac{10}{7}=\\boxed{10}$.
The answer is 10

Question: The difference between two numbers is 9, and the sum of the squares of each number is 153. What is the value of the product of the two numbers?
Let's think step by step
Call the first number $x$ and the second number $y$. 
Without loss of generality, assume $x > y$. 
We can represent the information given in the problem with the following system of linear equations:
\\begin{align*}
x - y &= 9\\\\
x^2 + y^2 &= 153
\end{align*} 
Solving for $x$ in the first equation and substituting into the second yields $(9+y)^2 + y^2 = 153$, or $2y^2 + 18y - 72 = 0$. 
Canceling a $2$ gives $y^2 + 9y - 36 = 0$, which factors into $(y+12)(y-3)$. 
Thus, $y = 3$ and $x = 12$. So, $x \cdot y = \\boxed{36}$.
The answer is 36

Question: Simplify $\\frac{1}{1+\sqrt{2}}\cdot\\frac{1}{1-\sqrt{2}}$.
Let's think step by step
Multiplying the numerators simply yields $1$. 
Multiplying the denominators gives $1+\sqrt{2} - \sqrt{2} -2 = 1 - 2 = -1$. 
So, the answer is $\\frac{1}{-1} = \\boxed{-1}$.
The answer is -1

Question: Four people can mow a lawn in 6 hours. How many more people will be needed to mow the lawn in 4 hours, assuming each person mows at the same rate?
Let's think step by step
The number of people mowing and the time required to mow are inversely proportional. 
Letting $n$ be the number of people and $t$ be the amount of time, we have $nt = (4)(6)= 24$ 
because 4 people can mow a lawn in 6 hours. 
If $m$ people can mow the lawn in 4 hours, then we must have $m(4) = 24$, so $m=6$.  
Therefore, we need $6-4 = \\boxed{2}$ more people to complete the job in 4 hours.
The answer is 2

Question: BoatsRUs built 7 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatsRUs by the end of May of this year?
Let's think step by step
The numbers of canoes built by BoatsRUs each month form a geometric sequence: 7, 14, 28, 56, 112. 
The first term is 7 and the common ratio is 2, so the sum of these terms is $\\frac{7(2^5-1)}{2-1} = \\boxed{217}$.
The answer is 217

Question: Find the coefficient of the $x^2$ term in the expansion of the product $(ax^3 + 3x^2 - 2x)(bx^2 - 7x - 4)$.
Let's think step by step
We only need to worry about the terms that multiply to have a degree of $2$. 
This would be given by the product of the terms $3x^2$ and $-4$ as well as the product of the terms $-2x$ and $-7x$. 
Thus, $$(3x^2) \\times (-4) + (-2x) \\times (-7x) = -12x^2 + 14x^2 = 2x^2,$$and the coefficient is $\\boxed{2}$.
The answer is 2'''

GSM8K = '''Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = \\boxed{6}.
The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step
There are originally 3 cars.
2 more cars arrive.
3 + 2 = \\boxed{5}.
The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let's think step by step
Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = \\boxed{39}.
The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step
Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = \\boxed{8}.
The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step
Shawn started with 5 toys.
If he got 2 toys each from his mom and dad, then that is 4 more toys.
5 + 4 = \\boxed{9}.
The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's think step by step
There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
9 + 20 is \\boxed{29}.
The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step
Michael started with 58 golf balls.
After losing 23 on tues- day, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = \\boxed{33} golf balls.
The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step
Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left.
23 - 15 is \\boxed{8}.
The answer is 8.'''

