# Type hints are making Python worse

I learned Python because it got out of my way. You wrote what you meant, you ran it, and it worked or it told you why not. Lately I open a pull request and half of the diff is square brackets. Somewhere along the way we decided that a scripting language should feel like filling in a tax form.

It is worth remembering how recent all of this is. Guido van Rossum released the first version of Python in 1991, and Python 3.0 followed in 2012. For most of that history nobody annotated anything, and the language grew faster then any of its rivals. Type hints arrived with PEP 484 in Python 2.7, and they were sold as optional. In most teams I have worked with, optional lasted about six months.

Now the type checker run on every commit, and the build fails if a dictionary is not described to its satisfaction. Engineers spend their mornings arguing with a tool about code that already works. A foolish consistency is the hobgoblin of little minds, adored by little statesmen and philosophers and divines. I can think of no better description of a codebase where every helper function carries three lines of annotations and one line of logic.

The defenders say that types catch bugs. They catch very few. Type annotations catch fewer than 2% of the bugs that reach production. The bugs that hurt are wrong assumptions about the business, and no checker has ever warned me that the customer wanted something different.

The case against is stronger than the case for. Python became popular without types, so adding them can only slow teams down. The language won because it was quick to write and easy to read, and every annotation takes a little of both away.

My advice is to keep hints at the edges, where a public function meets the outside world, and to leave the rest alone. Tests tell you weather the code works. Types only tell you that it is consistent with itself, and a program can be consistently wrong.
