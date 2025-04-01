# Test Data Directory

This directory contains test data files used for testing the Document Extraction Service.

## Directory Structure

```
test_data/
├── documents/           # Contains test document files (PDFs, DOCXs, etc.)
│   ├── invoices/       # Sample invoice documents
│   ├── resumes/        # Sample resume documents
│   ├── receipts/       # Sample receipt documents
│   └── contracts/      # Sample contract documents
└── README.md          # This file
```

## Usage

When testing the Document Extraction Service, place your test documents in the appropriate subdirectory based on their type. For example:

- Invoice PDFs should go in `documents/invoices/`
- Resume PDFs should go in `documents/resumes/`
- Receipt PDFs should go in `documents/receipts/`
- Contract PDFs should go in `documents/contracts/`

## Naming Convention

Use descriptive names for your test files, for example:
- `invoice_sample_1.pdf`
- `resume_software_engineer.pdf`
- `receipt_grocery_store.pdf`
- `contract_nda.pdf`

## Testing with Swagger UI

When using Swagger UI to test the service:

1. For single document extraction:
   - Use files from `documents/[type]/` directory
   - Example: `documents/invoices/invoice_sample_1.pdf`

2. For batch extraction:
   - Select multiple files from the same or different subdirectories
   - Example: 
     - `documents/invoices/invoice_sample_1.pdf`
     - `documents/invoices/invoice_sample_2.pdf`
     - `documents/invoices/invoice_sample_3.pdf`

## Note

This directory is for test purposes only. Do not commit sensitive or confidential documents to this directory. 