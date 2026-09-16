"""Small synthetic, hand-labeled email set used only for unit tests -- kept
deliberately unambiguous (distinct vocabulary per class) so that a passing
test reflects a genuinely working classifier, not a lucky guess."""
from ingestion import Email

SYNTHETIC_TRAIN = [
    Email("1", "Update my mailing address", "a@x.com", "2025-01-01",
          "Please update the address on my account to 12 Main Street.", "e1.html", "Account Management"),
    Email("2", "Change account username", "b@x.com", "2025-01-02",
          "I would like to change the username associated with my online banking account.", "e2.html", "Account Management"),
    Email("3", "Portfolio rebalancing question", "c@x.com", "2025-01-03",
          "Can my investment advisor rebalance my equity portfolio this quarter?", "e3.html", "Investment Advisory"),
    Email("4", "Mutual fund performance", "d@x.com", "2025-01-04",
          "What is the year to date return on the mutual fund in my investment account?", "e4.html", "Investment Advisory"),
    Email("5", "Mortgage loan application", "e@x.com", "2025-01-05",
          "I submitted a mortgage loan application last week, what is the status?", "e5.html", "Loan Processing"),
    Email("6", "Loan repayment schedule", "f@x.com", "2025-01-06",
          "Please send me the repayment schedule for my personal loan.", "e6.html", "Loan Processing"),
    Email("7", "Car accident insurance claim", "g@x.com", "2025-01-07",
          "I was in a car accident and need to file an insurance claim for the damage.", "e7.html", "Insurance Claims"),
    Email("8", "Insurance claim status", "h@x.com", "2025-01-08",
          "Following up on the status of my home insurance claim submitted last month.", "e8.html", "Insurance Claims"),
    Email("9", "Office holiday hours", "i@x.com", "2025-01-09",
          "What are your branch opening hours over the holiday period?", "e9.html", "Other"),
    Email("10", "Newsletter subscription", "j@x.com", "2025-01-10",
          "Please unsubscribe me from the monthly newsletter, it is unrelated to my accounts.", "e10.html", "Other"),
]

ALL_CATEGORIES = {
    "Account Management",
    "Investment Advisory",
    "Loan Processing",
    "Insurance Claims",
    "Other",
}
