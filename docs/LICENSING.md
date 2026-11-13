# Licensing: what you may do with Apsi

Apsi is published under the **Apsi License 2.0** (the `LICENSE` file). This page explains it in plain words with
examples. It is a guide, not the licence itself and not legal advice: if anything here differs from `LICENSE`, the
licence applies, and for a decision that matters to your business, ask a lawyer.

## The short version

* **Use Apsi for anything** — study, research, hobby, school, government, and commercial work in companies. Free.
* **Sell what you build with Apsi** — programs, libraries, apps, models, services. They are yours.
* **Do not sell Apsi itself** — no paid copies, paid downloads, or paid hosted Apsi without a licence from ATEC.
* **Credit ATEC** when you share Apsi or a changed version of it, and keep changed versions of Apsi open under the
  same licence.

## Allowed or not? Examples

| You want to… | Allowed? | Why |
|---|---|---|
| Learn Apsi, use it at university, teach a course with it | ✅ | any purpose is allowed (section 3.1) |
| Use Apsi at your company to build internal tools or analyse data | ✅ | use inside an organisation is not selling (section 2) |
| Write an app in Apsi and sell it, including as an executable from `apsi build` | ✅ | your programs are yours; the runtime may travel with them (sections 2, 6) |
| Run a paid web service whose back end is written in Apsi | ✅ | you are selling your program, not Apsi |
| Train an AI model with Apsi and sell the model or its predictions | ✅ | models and results belong to you (section 6) |
| Publish an Apsi library on apsi.dev under MIT, GPL, or a paid licence | ✅ | you choose the licence of your libraries (section 5.3) |
| Convert your company's Python code with `apsi convert` and keep it closed | ✅ | converted programs are your programs |
| Charge clients for consulting, development, training or support that uses Apsi | ✅ | charging for your own work is not selling the Software |
| Fork Apsi, improve it and share the fork for free | ✅ | keep the licence, credit ATEC, say what you changed, use your own name |
| Sell copies of Apsi or a fork of Apsi, or charge for downloading it | ❌ | that is Selling the Software (section 5.1) — ask ATEC for a licence |
| Offer a paid online Apsi compiler, notebook or IDE | ❌ | hosted Apsi where Apsi is what customers pay for |
| Build a paid language or compiler on top of Apsi's source code | ❌ | a Modified Version sold for money |
| Remove the credit or call your fork "Apsi" | ❌ | sections 5.2 and 5.4 |
| Put a fork under a different, more restrictive licence | ❌ | Modified Versions stay under this licence (section 5.3) |

Not sure? Ask yourself: *are people paying for **Apsi**, or for **what I made with it**?* The first needs a licence
from ATEC; the second is yours.

## What to write where

* **Sharing Apsi or a fork** — keep `LICENSE` and add this to your README:

  ```text
  Based on Apsi, originally developed by ATEC — https://github.com/ATECTechnologies/apsi
  Licensed under the Apsi License 2.0. Apsi itself may not be sold; programs written in Apsi are yours.
  ```

* **Shipping your program with the Apsi runtime** (for example an executable from `apsi build`) — one line in
  your documentation, licence file or "about" box:

  ```text
  Uses the Apsi runtime by ATEC (https://apsi.dev), Apsi License 2.0.
  ```

* **Programs and libraries you write in Apsi** — nothing is required. Use any licence you like.

## Contributions

Pull requests to the official repository are welcome. By sending one you confirm that you may contribute it, and you
allow ATEC to include it in Apsi under this licence and under ATEC's commercial licences, so that Apsi keeps one
licence. You keep the copyright of your work (section 8).

## Patents

ATEC grants a patent licence for using Apsi. It ends for anyone who sues claiming that Apsi infringes a patent
(section 4).

## If something goes wrong

If you break a condition, ATEC notifies you in writing and you have 30 days to fix it — for example by adding the
missing credit or taking down a paid offer. The first time, your rights come back as soon as you comply (section 9).

## Older versions

Apsi 1.10.0 and 2.0.0 were released under the Apsi Non-Commercial License 1.0. You may use them under the Apsi License
2.0 instead (section 13), which allows more.

## Sponsorship

Sponsoring ATEC through GitHub Sponsors is a voluntary gift handled by GitHub under its own terms. It does not change
the licence: Apsi is free under the Apsi License 2.0 for sponsors and non-sponsors alike, and a sponsorship does not
grant a commercial licence to sell Apsi itself.

## Why the licence looks like this

The licence was written after comparing the approaches that source-available projects use:

* **Non-commercial licences** such as the PolyForm Noncommercial License 1.0.0 allow only non-commercial use. That
  blocks companies from simply *using* the software, which is too narrow for a programming language. Apsi allows all
  use, and restricts only selling Apsi itself.
* **Delayed open-source licences** such as the Business Source License 1.1 restrict production use and convert to
  an open-source licence on a change date. Apsi does not restrict production use at all.
* **Patent clauses.** The patent licence and patent defence follow the common pattern in PolyForm and Apache-style
  licences.
* **Thai law.** The Copyright Act B.E. 2537 (1994) protects computer programs. Its exceptions (sections 32 and 35,
  for example study, research, back-up copies and adapting a program you lawfully use) stay available. Under the
  Unfair Contract Terms Act B.E. 2540 (1997), a party cannot exclude its own liability for fraud or gross
  negligence in advance, and courts can limit unfair terms. That is why the disclaimer applies "to the extent the
  applicable law allows" and leaves that liability in place.

## Sources

* [PolyForm Noncommercial License 1.0.0 (SPDX)](https://spdx.org/licenses/PolyForm-Noncommercial-1.0.0.html)
* [Business Source License 1.1 (SPDX)](https://spdx.org/licenses/BUSL-1.1.html)
* [Copyright Act B.E. 2537 (1994), Department of Intellectual Property, Thailand](https://ipthailand.go.th/images/26669/2566/laws/COPYRIGHT%20ACT%20B.E.%202537%20(1994).pdf)
* [Technology licensing in Thailand — Tilleke & Gibbins](https://www.tilleke.com/insights/technology-licensing-thailand)
