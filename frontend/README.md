# DeepGuard Frontend Architecture

The frontend of DeepGuard is designed as a modern, responsive web application served by the Flask application engine (`backend/app.py`).

## Technology Stack
- **Template Engine:** Jinja2 Template Inheritance
- **CSS Architecture:** Custom Design Token System (CSS variables in `tokens.css` & `theme.css`)
- **Icons & Visuals:** Inline SVG Icons & Dynamic Data Meters
- **Client Scripting:** Modern ES6 JavaScript (`backend/static/js/script.js`)

---

## Template Inventory (`backend/templates/`)

| Template | Route | Description |
|---|---|---|
| [`base.html`](../backend/templates/base.html) | — | Global master layout with navigation header, sidebar, flash alert system, and footer. |
| [`index.html`](../backend/templates/index.html) | `/` | Public landing page featuring platform hero, live feature breakdown, and quick CTA. |
| [`analyze.html`](../backend/templates/analyze.html) | `/analyze` | Primary multimodal submission interface with live word counter, drag-and-drop image upload, and modal progress meter. |
| [`result.html`](../backend/templates/result.html) | `/result/<id>` | Forensic breakdown report displaying verdict, confidence meter, risk level, EXIF table, viral heuristics, and XAI keyword chips. |
| [`dashboard.html`](../backend/templates/dashboard.html) | `/dashboard` | Authenticated user hub displaying total scan counts, authenticity distribution chart, recent scan feed, and shortcut cards. |
| [`history.html`](../backend/templates/history.html) | `/history` | Searchable and filterable archive of all historical scans submitted by the logged-in user. |
| [`login.html`](../backend/templates/login.html) | `/login` | Secure sign-in portal with validation and session authentication. |
| [`register.html`](../backend/templates/register.html) | `/register` | User onboarding portal with client & server password validation. |
| [`forgot_password.html`](../backend/templates/forgot_password.html) | `/forgot-password` | Account recovery request form generating cryptographic password reset tokens. |
| [`reset_password.html`](../backend/templates/reset_password.html) | `/reset-password/<token>` | Secure password change interface verifying matching credentials. |
| [`profile.html`](../backend/templates/profile.html) | `/profile` | User account settings, name/email update, and password modification forms. |
| [`modules.html`](../backend/templates/modules.html) | `/modules` | Technical architecture overview of all 7 AI detection modules. |
| [`about.html`](../backend/templates/about.html) | `/about` | Project scope, methodologies, team, and academic references. |

---

## Design System & Styling (`backend/static/css/`)

| Style Sheet | Role |
|---|---|
| [`tokens.css`](../backend/static/css/tokens.css) | Core design tokens (colors, typography scales, spacing, shadows, border radii). |
| [`theme.css`](../backend/static/css/theme.css) | Global base styles, card components, buttons, form controls, and navbar layout. |
| [`analyze.css`](../backend/static/css/analyze.css) | Drag-and-drop file upload target, character counters, and multimodal preview. |
| [`dashboard.css`](../backend/static/css/dashboard.css) | Metric counters, chart cards, recent scans list, and quick actions grid. |
| [`history.css`](../backend/static/css/history.css) | Historical scan data table, filter pills, and pagination. |
| [`modules.css`](../backend/static/css/modules.css) | Technical pipeline flowchart and module detail cards. |
| [`profile.css`](../backend/static/css/profile.css) | User settings card, security form, and account badges. |
| [`about.css`](../backend/static/css/about.css) | Project documentation layout and citation blocks. |
