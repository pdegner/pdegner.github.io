---
layout: page
permalink: /project-archive/
redirect_from:
  - /graduate-projects/
title: "Project Archive"
kicker: "Earlier work"
lede: >-
  These projects were completed during my graduate studies and represent earlier
  work in machine learning, statistics, visualization, and data engineering.
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
