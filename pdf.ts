import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';

// Configure worker to use local file
GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';

export async function extractTextFromPDF(pdfBuffer: ArrayBuffer): Promise<string> {
  try {
    const pdf = await getDocument({ data: pdfBuffer }).promise;
    let fullText = '';

    for (let i = 1; i <= pdf.numPages; i++) {
      try {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const pageText = content.items
          .filter((item: any) => typeof item.str === 'string')
          .map((item: any) => item.str.trim())
          .join(' ');
        
        // Add page marker at the start of each page's content
        fullText += `Page ${i}\n${pageText}\n\n`;
      } catch (pageError) {
        console.error(`Error extracting text from page ${i}:`, pageError);
        // Continue with next page if one fails
        fullText += `Page ${i}\nError extracting text from this page.\n\n`;
      }
    }

    if (!fullText.trim()) {
      throw new Error('No text could be extracted from the PDF. The file might be empty, corrupted, or password protected.');
    }

    return fullText.trim();
  } catch (error) {
    console.error('Error extracting text from PDF:', error);
    if (error instanceof Error) {
      throw new Error(`Failed to process PDF: ${error.message}`);
    }
    throw new Error('Failed to process PDF. Please make sure the file is not corrupted or password protected.');
  }
}