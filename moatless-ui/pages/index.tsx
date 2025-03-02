import { GenericContentView } from "../components/GenericContentView";
import type { ContentStructure } from "../types/content";

const verificationData: ContentStructure = {
  title: "Verification Data",
  sections: [
    {
      id: "main",
      title: "Verification Details",
      blocks: [
        {
          id: "status",
          type: "text",
          content: "Status: New",
          variant: "subheading",
        },
        {
          id: "description",
          type: "text",
          content: "Description: Anthropic - API Credits (2024-05-13)",
        },
        { id: "voucher", type: "text", content: "Voucher Series: A" },
        { id: "date", type: "text", content: "Accounting Date: 2024-05-13" },
      ],
      sections: [
        {
          id: "transactions",
          title: "Transactions",
          blocks: [
            {
              id: "transactions-table",
              type: "table",
              headers: [
                "Account",
                "Transaction Info",
                "Debit",
                "Credit",
                "Currency",
                "Original Amount",
              ],
              rows: [
                [
                  "5422",
                  "Anthropic - API Credits (2024-05-13) (USD 24.99 @ 10.8384)",
                  270.85,
                  0,
                  "USD",
                  24.99,
                ],
                [
                  "2645",
                  "Anthropic - API Credits (2024-05-13) (USD 6.25 @ 10.8384)",
                  67.71,
                  0,
                  "USD",
                  6.25,
                ],
                [
                  "2614",
                  "Anthropic - API Credits (2024-05-13) (USD 6.25 @ 10.8384)",
                  0,
                  67.71,
                  "USD",
                  6.25,
                ],
                [
                  "2820",
                  "Anthropic - API Credits (2024-05-13) (USD 24.99 @ 10.8384)",
                  0,
                  270.85,
                  "USD",
                  24.99,
                ],
              ],
            },
            { id: "total-debit", type: "text", content: "Total Debit: 338.56" },
            {
              id: "total-credit",
              type: "text",
              content: "Total Credit: 338.56",
            },
          ],
        },
      ],
    },
  ],
};

const receiptData: ContentStructure = {
  title: "Receipt Data",
  sections: [
    {
      id: "main",
      title: "Receipt Details",
      blocks: [
        {
          id: "status",
          type: "text",
          content: "Status: New",
          variant: "subheading",
        },
        { id: "merchant", type: "text", content: "Merchant: Anthropic" },
        { id: "country", type: "text", content: "Receipt Country: US" },
        { id: "amount", type: "text", content: "Total Amount: $24.99" },
      ],
      sections: [
        {
          id: "items",
          title: "Items",
          blocks: [
            {
              id: "items-table",
              type: "table",
              headers: ["Name", "Quantity", "Unit Price", "Total Price"],
              rows: [["One-time credit purchase", 1, 25, 25]],
            },
          ],
        },
        {
          id: "merchant-contact",
          title: "Merchant Contact",
          blocks: [
            {
              id: "contact-list",
              type: "list",
              items: [
                "Email: support@anthropic.com",
                "Phone: N/A",
                "Website: N/A",
                "Org Number: N/A",
              ],
            },
          ],
        },
        {
          id: "additional-info",
          title: "Additional Information",
          blocks: [
            {
              id: "receipt-number",
              type: "text",
              content: "Receipt Number: 2848-8242",
            },
            {
              id: "purchase-date",
              type: "text",
              content: "Purchase Date: 2024-05-13",
            },
            {
              id: "payment-method",
              type: "text",
              content: "Payment Method: American Express - 1009",
            },
          ],
        },
        {
          id: "vat-details",
          title: "VAT Details",
          blocks: [
            {
              id: "vat-table",
              type: "table",
              headers: ["Rate", "Net Amount", "VAT Amount"],
              rows: [[0, 25, 0]],
            },
          ],
        },
        {
          id: "additional-details",
          title: "Additional Details",
          blocks: [
            { id: "currency", type: "text", content: "Currency: USD" },
            {
              id: "account-number",
              type: "text",
              content: "Account Number: 5422",
            },
          ],
        },
      ],
    },
  ],
};

export default function Home() {
  return (
    <div className="p-8">
      <h1 className="text-4xl font-bold mb-8">Generic Content View Examples</h1>
      <div className="space-y-12">
        <GenericContentView content={verificationData} />
        <GenericContentView content={receiptData} />
      </div>
    </div>
  );
}
