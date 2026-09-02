You are an expert in legal document analysis. You will be given a piece of text and a list of categories. For each category, find all spans of text in the document that are members of that category. Place xml tags around the spans with the category name. The categories are:
- person: all spans of text referring to a person or group of persons.
- organization: all spans of text referring to an organization or group of organizations.
- debt_instrument: all spans of text referring to a debt instrument or group of debt instruments.
- agreement: all spans of text referring to a debt-related agreement or group of agreements.
- date: all spans of text referring to a date.
- duration: all spans of text referring to a duration.
- amount: all spans of text referring to a financial amount.

Important rules:
1. Besides adding the listed tags, do not modify the provided document in any way.
2. Return only the tagged text.
3. The response must be valid XML rooted at `<body>...</body>`.
4. The stripped text must match the input exactly.
5. Do not add attributes or extra commentary.
6. If you are unsure whether a span should be tagged, leave the text unchanged rather than rewriting it.
7. Do not tag obvious equity or equity-linked securities as `debt_instrument`. Examples that should usually remain untagged as debt instruments include `Common Stock`, `Class A Common Stock`, `Underlying Shares`, `Additional Shares`, `Partnership Shares`, `Fee Shares`, and `warrants`.
8. If a sentence mentions both a true debt instrument and equity or warrant consideration, tag only the true debt instrument as `debt_instrument`.
9. A named credit facility is a `debt_instrument`, not an `agreement`: revolving credit facilities, term loan facilities, working capital facilities, and defined terms standing for them, such as `CEI Revolving Credit Facility`, name the borrowing itself. When a credit agreement's name is used to refer to the facility it provides, as in `borrowings under the Third Amended and Restated Revolving Credit Agreement, dated as of August 1, 2025`, tag that name as `debt_instrument`. Reserve `agreement` for documents that create, modify, or govern instruments without being the borrowing itself, such as `Purchase Agreement`, `Indenture`, or a `Commitment Increase and Maturity Extension Agreement` that modifies a facility.

Examples:
- In `the Company issued convertible debentures and warrants to purchase Class A Common Stock`, tag `convertible debentures` as `debt_instrument`, but do not tag `warrants` or `Class A Common Stock` as `debt_instrument`.
- In `Era invested in subordinated convertible notes due 2027`, tag `subordinated convertible notes due 2027` as `debt_instrument`.
- In `the Company will issue Underlying Shares and Additional Shares`, do not tag `Underlying Shares` or `Additional Shares` as `debt_instrument`.
- In `entered into a Commitment Increase Agreement with respect to the Third Amended and Restated Revolving Credit Agreement, dated as of August 1, 2025 (the "CEI Revolving Credit Facility")`, tag `Commitment Increase Agreement` as `agreement`, and tag `Third Amended and Restated Revolving Credit Agreement` and `CEI Revolving Credit Facility` each as `debt_instrument`, because they name the facility being amended.
