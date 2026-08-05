---
layout: page
permalink: /project-archive/
redirect_from:
  - /graduate-projects/
title: "Project Archive"
kicker: "Earlier work"
lede: >-
  These projects were completed during my graduate studies and represent earlier work in machine learning, statistics, visualization, and data engineering. Relics from the olden days, when code was written by hand and, BERT was the hot new language model, and if you had a question, your best option was to descend into the bowels of Stack Overflow—or worse: read the documentation.

  Every line, comma, and deeply nested parenthesis on this page was carefully authored by me, with no assistance from AI.

  Modern AI tools have rendered much of this work obsolete. Still, I look back on it fondly. Building it forced me to understand the code, the modeling decisions, and the mechanics of natural language processing in a way that is much harder to replicate when an AI assistant can generate the solution in seconds.
description: >-
  Machine-learning, statistics and data-engineering projects from Patti Degner's
  UC Berkeley MIDS program.
---

<ul class="archive-list">
{%- for project in site.data.projects.archive %}
  {% include archive-card.html project=project %}
{%- endfor %}
</ul>

<p class="doc-back" style="margin-top: var(--space-xl)">
  <a href="{{ '/#projects' | relative_url }}">&larr; Back to featured projects</a>
</p>
