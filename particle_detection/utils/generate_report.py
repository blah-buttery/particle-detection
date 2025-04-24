from fpdf import FPDF
import os

class PDFReport(FPDF):
    """Custom PDF report class for nanoparticle detection evaluation.

    This class extends the FPDF library to provide a custom header and footer
    for the evaluation report.
    """

    def header(self):
        """Defines the header for each page of the PDF report."""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Nanoparticle Detection Evaluation Report', ln=True, align='C')

    def footer(self):
        """Defines the footer for each page of the PDF report, including the page number."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


def create_pdf_report(save_dir, text_results):
    """Generates a PDF evaluation report summarizing detection results and embedding result images.

    The report includes:
        - A summary section listing the number of particles detected per test image.
        - Embedded images for each test sample, including k-distance plots, clustering visualizations,
          overlays, and cluster labeling results.

    Args:
        save_dir (str): Path to the directory where the output PDF report and related images are saved.
        text_results (list of tuple): List of tuples where each tuple contains:
            - int: Image index (e.g., test image number).
            - dict: A result dictionary with at least the key `'num_particles'` specifying the number
              of detected particles for that image.

    Example:
        text_results = [
            (0, {"num_particles": 15}),
            (1, {"num_particles": 22})
        ]

    The function searches for the following image files for each test image index:
        - k_distance_{index}.png
        - cluster_view_{index}.png
        - visualize_clusters_{index}.png
        - label_clusters_{index}.png
        - overlay_{index}.png

    If these images exist, they will be added to the report on separate pages.

    Saves:
        The final PDF report as `evaluation_report.pdf` inside `save_dir`.

    Prints:
        A confirmation message with the full path to the generated report.

    Raises:
        None.
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, "Evaluation Summary:", ln=True)

    for i, result in text_results:
        pdf.multi_cell(0, 10,
            f"Test Image {i}\n"
            f"Particles Detected: {result['num_particles']}\n"
        )
        pdf.ln(5)

    # Add associated images
    for i, _ in text_results:
        for img_file in ["k_distance", "cluster_view", "visualize_clusters", "label_clusters", "overlay"]:
            img_path = os.path.join(save_dir, f"{img_file}_{i}.png")
            if os.path.exists(img_path):
                pdf.add_page()
                pdf.cell(0, 10, f"Image: {img_file}_{i}", ln=True)
                pdf.image(img_path, x=10, y=30, w=180)

    report_path = os.path.join(save_dir, "evaluation_report.pdf")
    pdf.output(report_path)
    print(f"PDF report generated at: {report_path}")
