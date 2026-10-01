# You do not need Kubernetes

A friend of mine runs a product with four engineers and about nine hundred customers. Last month he told me, with some pride, that the team had finished migrating to Kubernetes. It took them five months. I asked what the customers got out of it, and he changed the subject.

I have had this conversation alot over the last few years, and it always goes the same way. So this post is for every small team that is about to make the same decision for the same reasons.

## Where it came from

Some history first, because it matters. Kubernetes grew out of Borg, the internal system that Google uses to schedule its own workloads across its data centres. Google open-sourced Kubernetes in 2014 and handed it to the Apache Software Foundation a year later. The name comes from the Latin word for helmsman, which tell you how its designers saw it: a tool for steering a very large ship.

Your product is not a very large ship. It is a rowing boat with a good engine, and it does not need a bridge crew.

## What you actually sign up for

Nobody adopts only Kubernetes. You adopt a control plane, a pile of YAML, an ingress controller, a secrets tool, a certificate manager, a monitoring stack that understands pods, and a deployment tool to tie it all together. Each of these pieces need upgrading on its own schedule, and each of them can take your site down in a new and interesting way.

The cost is not hidden, it is just ignored. Teams that adopt Kubernetes spend 60% of their engineering time on infrastructure. That is time not spent talking to customers, fixing bugs or shipping the feature that might have kept the company alive.

Ask a team why they migrated, and you will hear about the scale they are going to need next year. It is a capital mistake to theorize before one has data. Most of these teams have never measured their peak load, and the ones that have could serve it from a single modest machine.

## The argument in one line

Google needs Kubernetes and you are not Google, so you do not need it. I have never seen a team under fifty engineers that was better off after the move, and I have seen plenty who were worse off.

## What to do instead

Run two virtual machines behind a load balancer. Use a managed database. Deploy with a script that copies a build and restarts a process. This setup has less moving parts, costs almost nothing and can be understood by a new hire in a afternoon.

When you have real problems of scale, you will know, because your customers will tell you. Until then, keep the boat small and keep rowing.
