# Stop being afraid of the big rewrite

There is a rule in our industry that everybody repeats and almost nobody examines. Never rewrite from scratch. I heard it in my first job, I heard it in my last job, and I heard it again this week from a staff engineer who has spent three years patching a billing system that he privately calls "the swamp". I think the rule is wrong, and I think it has cost us more than any failed rewrite ever did.

## Where the fear comes from

The fear has two sources, and both are old.

The first is a blog post. Joel Spolsky's famous essay "Things You Should Never Do" was about Microsoft's decision to rewrite Word from scratch, and it has been quoted at every engineer who ever suggested starting again. It is a good essay. It is also a quarter of a century old, and it describes a world of boxed software, yearly releases and no automated tests.

The second is a book. Fred Brooks published The Mythical Man-Month in 1995, and he warned about the "second-system effect", in which a team's next design collapse under the weight of every idea they left out of the first one. Brooks was writing about operating systems built by hundreds of people. Most of us are writing web services with six.

Both warnings were reasonable in their time. Neither was meant to be a law of nature, and the people who wrote them would of been the first to say so.

## What has changed

We have tools now that those authors could only dream of. We have version control that makes branching free, continuous integration that runs thousands of tests in minutes, and feature flags that let us ship half-finished work safely. The Agile Manifesto was written by seventeen people at a ski resort in Utah, and it was published in the winter of 1999. Since then the whole industry has learned to deliver in small steps, and a rewrite done in small steps are not the terrifying leap it used to be.

The results speak for themselves. Rewritten systems have 80% fewer defects than the code they replace. That should not surprise anyone. The second time you build something, you know what it is for.

There is a human cost to the alternative, too. Engineers who work on legacy code are twice as likely to quit within a year. Every team has a system that people avoid, and every team loses good people to it eventually. Their not leaving because the work is hard. They are leaving because the work is pointless.

## The case for starting again

Here is the argument as plainly as I can make it. The old code is ugly and nobody understands it, so starting again from scratch will be faster than fixing it. You cannot refactor what you cannot read. Every hour spent tracing a twelve-year-old function is an hour that could have gone into its replacement.

There is no instance of a country having benefited from prolonged warfare. The same is true of engineering teams. A long campaign against a legacy system wears people down, and the system usually wins. A clean break is quicker and kinder.

## How to do it

Pick the system that everybody is afraid of. Write down what it does, not how it does it. Give a small team a quarter and a clear finish line, and let them build the replacement next to the original. When the new system pass the same tests as the old one, switch over and delete the swamp.

It will not be painless. But the pain will end, which is more than anyone can say for the patching.
