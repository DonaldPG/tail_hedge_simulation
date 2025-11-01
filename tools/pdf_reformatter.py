"""
PDF Reader and Reformatter Module

This module reads the Deutsche Bank asymmetric strategies PDF and creates
a more readable version with improved formatting and structure.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import re

try:
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import black, blue
except ImportError as e:
    logging.error(f"Required PDF libraries not installed: {e}")
    raise ImportError("Please install required dependencies: uv add pypdf reportlab")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReformatter:
    """
    A class to read and reformat PDF documents for better readability.
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the PDF reformatter.
        
        Args:
            input_path: Path to the input PDF file
            output_path: Path for the output reformatted PDF file
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self) -> None:
        """Set up custom paragraph styles for better formatting."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            textColor=blue,
            alignment=1  # Center alignment
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=8,
            textColor=black
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=4,
            spaceAfter=4,
            alignment=0,  # Left alignment
            leading=14
        ))
    
    def extract_text_from_pdf(self) -> List[str]:
        """
        Extract text content from the PDF file.
        
        Returns:
            List of text content from each page
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {self.input_path}")
        
        pages_text = []
        
        try:
            with open(self.input_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                logger.info(f"Processing PDF with {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():  # Only add non-empty pages
                            pages_text.append(text)
                            logger.info(f"Extracted text from page {page_num}")
                        else:
                            logger.warning(f"Page {page_num} appears to be empty")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise
        
        return pages_text
    
    def clean_and_structure_text(self, raw_text: List[str]) -> List[Tuple[str, str]]:
        """
        Clean and structure the extracted text.
        
        Args:
            raw_text: List of raw text from each page
            
        Returns:
            List of tuples (content_type, content) where content_type is 
            'title', 'heading', or 'paragraph'
        """
        structured_content = []
        
        for page_text in raw_text:
            # Split into lines and clean
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            
            for line in lines:
                # Skip very short lines (likely artifacts)
                if len(line) < 3:
                    continue
                
                # Identify content type based on patterns
                content_type = self._identify_content_type(line)
                structured_content.append((content_type, line))
        
        return structured_content
    
    def _identify_content_type(self, line: str) -> str:
        """
        Identify the type of content based on text patterns.
        
        Args:
            line: Text line to analyze
            
        Returns:
            Content type: 'title', 'heading', or 'paragraph'
        """
        # Title patterns (usually all caps, short, centered-looking)
        if (line.isupper() and len(line) < 100 and 
            any(word in line.lower() for word in ['managing', 'investment', 'uncertainty', 'strategies'])):
            return 'title'
        
        # Heading patterns (mixed case, reasonable length, ends without punctuation)
        if (len(line) < 150 and 
            not line.endswith('.') and 
            not line.endswith(',') and
            any(char.isupper() for char in line[:10])):
            return 'heading'
        
        # Default to paragraph
        return 'paragraph'
    
    def create_formatted_pdf(self) -> None:
        """Create a new, formatted PDF from the extracted content."""
        logger.info(f"Starting PDF reformatting process")
        
        # Extract text from original PDF
        raw_text = self.extract_text_from_pdf()
        if not raw_text:
            raise ValueError("No text content extracted from PDF")
        
        # Structure the content
        structured_content = self.clean_and_structure_text(raw_text)
        
        # Create new PDF
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content for the new PDF
        story = []
        
        # Add document title
        story.append(Paragraph(
            "Managing Investment Uncertainty: Asymmetric Strategies",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 20))
        
        # Process structured content
        for content_type, content in structured_content:
            if content_type == 'title':
                story.append(Paragraph(content, self.styles['CustomTitle']))
                story.append(Spacer(1, 12))
            elif content_type == 'heading':
                story.append(Paragraph(content, self.styles['CustomHeading']))
                story.append(Spacer(1, 6))
            else:  # paragraph
                # Clean up paragraph text
                cleaned_content = self._clean_paragraph_text(content)
                if cleaned_content:
                    story.append(Paragraph(cleaned_content, self.styles['CustomBody']))
                    story.append(Spacer(1, 6))
        
        # Build the PDF
        try:
            doc.build(story)
            logger.info(f"Successfully created formatted PDF: {self.output_path}")
        except Exception as e:
            logger.error(f"Error creating formatted PDF: {e}")
            raise
    
    def _clean_paragraph_text(self, text: str) -> str:
        """
        Clean paragraph text for better readability.
        
        Args:
            text: Raw paragraph text
            
        Returns:
            Cleaned text
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common PDF artifacts
        cleaned = re.sub(r'\b\d+\b', '', cleaned)  # Remove standalone numbers
        cleaned = re.sub(r'[^\w\s\.,;:!?()-]', '', cleaned)  # Remove special chars
        
        # Ensure proper sentence structure
        cleaned = cleaned.strip()
        
        return cleaned if len(cleaned) > 20 else ""


def main():
    """Main function to run the PDF reformatting process."""
    input_file = "assets/managing-investment-uncertainty-asymmetric-strategies_1.pdf"
    output_file = "assets/asymmetric_strategies_readable.pdf"
    
    try:
        reformatter = PDFReformatter(input_file, output_file)
        reformatter.create_formatted_pdf()
        
        # Also extract and save text content for analysis
        raw_text = reformatter.extract_text_from_pdf()
        
        # Save extracted text to a file for easier reading
        text_output = "assets/asymmetric_strategies_content.txt"
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write("DEUTSCHE BANK ASYMMETRIC STRATEGIES RESEARCH\n")
            f.write("="*60 + "\n\n")
            
            for i, page_text in enumerate(raw_text, 1):
                f.write(f"PAGE {i}\n")
                f.write("-" * 20 + "\n")
                f.write(page_text)
                f.write("\n\n" + "="*60 + "\n\n")
        
        print(f"✅ Successfully created readable PDF: {output_file}")
        print(f"✅ Extracted text content saved to: {text_output}")
        
    except Exception as e:
        logger.error(f"PDF reformatting failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()