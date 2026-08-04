# pdegner.github.io

Source for [pdegner.github.io](https://pdegner.github.io) — my portfolio.

Plain Jekyll on GitHub Pages. No theme gem, no CSS framework, no build step for
the styles, and no JavaScript beyond a ~25-line theme toggle.

## Layout of the repo

```
_config.yml          site metadata, nav, SEO defaults
_data/
  projects.yml       featured projects + the archive list
  experience.yml     selected professional work
  skills.yml         grouped skills
_layouts/
  base.html          document shell
  home.html          the homepage, assembled from includes
  page.html          standard content page
  archive-doc.html   wrapper for the imported graduate write-ups
_includes/           head, header, footer, and the reusable card components
assets/
  css/main.css       the whole design system: tokens first, then components
  img/               headshot, favicon, Open Graph card
index.md             front matter only — content lives in _data + _layouts
project-archive.md   renders _data/projects.yml `archive:`
Machine_Learning/, Python/, R/   graduate-school write-ups (archive)
```

Content is kept out of the markup: to add or edit a project, change
`_data/projects.yml`. `_includes/project-card.html` renders whatever is there.

## Styling

`assets/css/main.css` is plain CSS, served as-is. Colors, spacing, type scale,
radii and layout widths are all custom properties declared once at the top.

Both themes are defined in a single token block using `light-dark()`, so there
is no duplicated dark-mode stylesheet. The page follows the visitor's OS setting
by default; the header toggle writes an explicit choice to `localStorage` and
overrides `color-scheme` on `:root`. To ship dark as the default instead, set
`color-scheme: dark light` on `:root`.

## Running it locally

```sh
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>.

## License

Code is MIT (see `LICENSE`). Project write-ups and images are not.
