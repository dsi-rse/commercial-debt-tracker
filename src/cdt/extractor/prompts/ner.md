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

Examples:
- In `the Company issued convertible debentures and warrants to purchase Class A Common Stock`, tag `convertible debentures` as `debt_instrument`, but do not tag `warrants` or `Class A Common Stock` as `debt_instrument`.
- In `Era invested in subordinated convertible notes due 2027`, tag `subordinated convertible notes due 2027` as `debt_instrument`.
- In `the Company will issue Underlying Shares and Additional Shares`, do not tag `Underlying Shares` or `Additional Shares` as `debt_instrument`.
