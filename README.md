*For our [ECE 533]: Digital Image Processing final project*

### Development
Let's have all development take place in Python. This also allows for access to
the [Jupyter notebook], which is a great way to share results. There's an
example in the file Project Proposal.ipynb.

[Jupyter notebook]:https://jupyter.org

I detail how to step from Matlab to Python in a [blog post] -- it's not a big
switch but offers some nice benefits.

Here's some sample code that shows how similar they are:

```python
from pylab import *

x = linspace(0, 1)
y = x**2

M, N = 3, 4
A = rand(M, N)
assert rank(A) == min(M, N)

figure()
plot(x, y)
show()
```

### Collaboration
Let's use GitHub to collaborate. This is how I develop; it allows to revert to
previous versions and provides a framework for collaboration. It's pretty easy
to get start with [GitHub Desktop].

[GitHub Desktop]:https://desktop.github.com
[blog post]:http://scottsievert.github.io/blog/2015/09/01/matlab-to-python/
[ECE 533]:http://courses.engr.wisc.edu/ece/ece533.html
