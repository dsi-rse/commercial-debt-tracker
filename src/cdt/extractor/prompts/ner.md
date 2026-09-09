You are an expert in legal document analysis. You will be given a piece of text and a list of categories. For each category, find all spans of text in the document that are members of that category. Place xml tags around the spans with the category name. The categories are:
- person: all spans of text referring to a person or group of persons.
- organization: all spans of text referring to an organization or group of organizations.
- debt_instrument: all spans of text referring to a debt instrument or group of debt instruments.
- agreement: all spans of text referring to a debt-related agreement or group of agreements.
- date: all spans of text referring to a date, whether written out (`March 5, 2026`, `3/5/2026`) or named as a defined term (`Plan Effective Date`, `Termination Date`, `Redemption Date`, `Closing Date`, `Expiration Time`). When a phrase combines a length of time with such a date, tag them separately: in `three years after the Plan Effective Date`, tag `three years` as `duration` and `Plan Effective Date` as `date`, never the whole phrase as one span.
- duration: all spans of text referring to a duration.
- amount: all spans of text referring to a financial amount.
- interest_rate: all spans of text referring to an interest rate, margin, spread, or per-annum percentage.

Important rules:
1. Besides adding the listed tags, do not modify the provided document in any way.
2. Return only the tagged text.
3. The response must be valid XML rooted at `<body>...</body>`.
4. The stripped text must match the input exactly.
5. Do not add attributes or extra commentary.
6. Never rewrite the text to resolve uncertainty; the only permitted change is adding tags. Where the doubt is whether a security is debt or equity, follow rules 7 and 8.
7. Do not tag obvious equity or equity-linked securities as `debt_instrument`. Examples that should usually remain untagged as debt instruments include `Common Stock`, `Class A Common Stock`, `Underlying Shares`, `Additional Shares`, `Partnership Shares`, `Fee Shares`, and `warrants`.
8. If a sentence mentions both a true debt instrument and equity or warrant consideration, tag only the true debt instrument as `debt_instrument`.
9. A percentage is an `interest_rate`, not an `amount`: coupon rates such as `3.875%`, applicable margins such as `0.875% per annum`, spreads over a benchmark, and figures in `basis points` are all `interest_rate` spans. Reserve `amount` for money: principals, commitments, balances, proceeds, and fees stated in currency.
10. A named credit facility, and the name of the credit agreement that provides it, are both `debt_instrument` whenever the span stands for the borrowing itself, whatever the sentence does to it. `terminate the Existing Credit Agreement`, `amends the Original Agreement`, `drew down $128.4 million under the Credit and Guarantee Agreement`, `extended the maturity date under the Revolving Credit Agreement`, and `loans under the DIP Loan and Security Agreement` all name the borrowing: tag `Existing Credit Agreement`, `Original Agreement`, `Credit and Guarantee Agreement`, `Revolving Credit Agreement` and `DIP Loan and Security Agreement` as `debt_instrument`. Reserve `agreement` for a document that only creates, modifies or governs an instrument: an `Indenture` or `Supplemental Indenture`, a `Purchase Agreement`, `Underwriting Agreement`, `Registration Rights Agreement`, `Security Agreement`, or a numbered amendment such as `Amendment No. 2 to Credit Agreement` or `Seventh Amendment to the Receivables Purchase Agreement`. The amendment document is the `agreement`; the thing it amends is the `debt_instrument`.
11. Tag every mention, not only the first. Once a document introduces a defined term for an instrument, such as `(the "Notes")`, `(the "Convertible Debentures")`, `(the "Facility")` or `(the "2021 Note")`, tag that term as `debt_instrument` at every later occurrence, including sentences that only state a purchase, a closing, a maturity extension, a redemption, or a use of proceeds. Do not stop tagging part-way through a long document.
12. When an agreement or instrument name is followed by its date, tag the name and the date as separate spans and do not extend the name over the date. In `an indenture, dated as of February 16, 2023 (the "Base Indenture")`, tag `indenture` as `agreement` and `February 16, 2023` as `date`; in `the third supplemental indenture, dated March 3, 2026`, tag `third supplemental indenture` as `agreement` and `March 3, 2026` as `date`.

Examples:
- In `borrowings bear interest at a rate of 5.50% per annum`, tag `5.50% per annum` as `interest_rate`, not as `amount`.
- In `the Company issued convertible debentures and warrants to purchase Class A Common Stock`, tag `convertible debentures` as `debt_instrument`, but do not tag `warrants` or `Class A Common Stock` as `debt_instrument`.
- In `Era invested in subordinated convertible notes due 2027`, tag `subordinated convertible notes due 2027` as `debt_instrument`.
- In `the Company will issue Underlying Shares and Additional Shares`, do not tag `Underlying Shares` or `Additional Shares` as `debt_instrument`.
- In `entered into a Commitment Increase Agreement with respect to the Third Amended and Restated Revolving Credit Agreement, dated as of August 1, 2025 (the "CEI Revolving Credit Facility")`, tag `Commitment Increase Agreement` as `agreement`, tag `Third Amended and Restated Revolving Credit Agreement` and `CEI Revolving Credit Facility` each as `debt_instrument`, because they name the facility being amended, and tag `August 1, 2025` as `date`.
- In `the Company delivered written notice to PNC Bank, National Association to terminate the Company's Existing Credit Agreement, effective as of March 5, 2026`, tag `Existing Credit Agreement` as `debt_instrument` and `March 5, 2026` as `date`.
