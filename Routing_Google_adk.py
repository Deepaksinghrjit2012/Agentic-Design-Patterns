import asyncio
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

load_dotenv()
def create_job_description(request: str) -> str:
    """
    Prepare a job description brief that the LLM should expand into a full, descriptive posting.

    Use this tool when the user provides a brief overview of the job role, responsibilities, and requirements.
    The output should preserve the input fields but add explicit instructions for the model to write
    a complete, descriptive, ATS-friendly job description with sections and bullet points.
    """
    lines = [line.strip() for line in request.splitlines() if line.strip()]
    prompt = [
        "[JOB_DESCRIPTION]",
        "Create a professional, descriptive, ATS-friendly job description in markdown.",
        "Expand the user-provided notes into a polished posting with a clear title, summary, responsibilities, qualifications, and benefits if relevant.",
        "Do not copy the input verbatim. Write the full content yourself.",
        "",
        "User input:",
        *lines,
    ]
    return "\n".join(prompt)
def job_description_to_md(request: str) -> str:
    """
    Normalize model-generated job description content into markdown-friendly structure with clear headings and emphasized fields.

    Use this tool after job description content has been generated and needs to be normalized into markdown
    before it is passed to the PDF renderer.
    """
    lines = [line.rstrip() for line in request.splitlines() if line.strip()]
    md_lines = []
    for line in lines:
        if line.startswith("# "):
            md_lines.append(line)
        elif line.startswith("## "):
            md_lines.append(line)
        elif line.startswith("- "):
            md_lines.append(line)
        elif ":" in line and not line.startswith("http"):
            key, value = line.split(":", 1)
            if key.lower() in {"job title", "location", "key responsibilities"}:
                md_lines.append(f"**{key.strip()}:** {value.strip()}")
            else:
                md_lines.append(line)
        else:
            md_lines.append(line)
    return "\n".join(md_lines)


def resume_builder_content(request: str) -> str:
    """
    Clean and normalize the user's resume input before the LLM expands it into full resume content.

    Use this tool when the input is a short set of resume anchors such as:
    first name, last name, phone, email, location, and key skills.
    """
    lines = [line.strip() for line in request.splitlines() if line.strip()]
    return "\n".join(["[RESUME]", *lines])


def convert_resume_to_md(request: str) -> str:
    """
    Convert resume content into markdown-friendly structure with clear headings and emphasized fields.

    Use this tool after resume content has been generated and needs to be normalized into markdown
    before it is passed to the PDF renderer.
    """
    lines = [line.rstrip() for line in request.splitlines() if line.strip()]
    md_lines = []
    for line in lines:
        if line.startswith("# "):
            md_lines.append(line)
        elif line.startswith("## "):
            md_lines.append(line)
        elif line.startswith("- "):
            md_lines.append(line)
        elif ":" in line and not line.startswith("http"):
            key, value = line.split(":", 1)
            if key.lower() in {"first name", "last name", "phone", "key skills"}:
                md_lines.append(f"**{key.strip()}:** {value.strip()}")
            else:
                md_lines.append(line)
        else:
            md_lines.append(line)
    return "\n".join(md_lines)


def convert_md_to_pdf(request: str) -> str:
    """
    Prepare markdown resume text for PDF rendering by preserving headings, bullets, and section flow.

    Use this tool when markdown content is ready and should be passed to the PDF generation step.
    """
    lines = [line.strip() for line in request.splitlines() if line.strip()]
    pdf_ready = []
    for line in lines:
        if line.startswith("**") and line.endswith("**"):
            pdf_ready.append(line.strip("*"))
        else:
            pdf_ready.append(line)
    return "\n".join(pdf_ready)


def save_pdf_to_local_folder(request: str) -> str:
    """
    Render the final document text into a PDF file and save it in the current project folder.

    Use this tool after a resume or job description has been fully written in markdown and is ready to be exported.
    """
    base_dir = Path(__file__).resolve().parent
    normalized = request.lower()
    if "[job_description]" in normalized or "job description" in normalized or "job title" in normalized:
        base_name = "job_description"
    elif "[resume]" in normalized:
        base_name = "resume"
    else:
        base_name = "resume"

    md_path = base_dir / f"{base_name}.md"
    output_path = base_dir / f"{base_name}.pdf"
    md_path.write_text(request, encoding="utf-8")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ResumeTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=10,
    )
    header_style = ParagraphStyle(
        "ResumeHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#111827"),
        spaceBefore=10,
        spaceAfter=4,
        underlineWidth=1,
    )
    body_style = ParagraphStyle(
        "ResumeBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        "ResumeBullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=0,
    )

    story = []
    lines = [line.rstrip() for line in request.splitlines() if line.strip()]
    if not lines:
        lines = ["# Resume"]
    lines = [line for line in lines if line not in {"[RESUME]", "[JOB_DESCRIPTION]"}]

    for line in lines:
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), title_style))
            story.append(Spacer(1, 0.12 * inch))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), header_style))
        elif line.startswith("- "):
            story.append(Paragraph(line[2:].strip(), bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    return f"Saved {md_path.name} and {output_path.name} to the project folder."


resume_builder_content_tool = FunctionTool(resume_builder_content)
convert_resume_to_md_tool = FunctionTool(convert_resume_to_md)
convert_md_to_pdf_tool = FunctionTool(convert_md_to_pdf)
save_pdf_to_local_folder_tool = FunctionTool(save_pdf_to_local_folder)
create_job_description_tool = FunctionTool(create_job_description)
convert_job_description_to_md_tool = FunctionTool(job_description_to_md)


resume_builder_agent = Agent(
    name="resume_builder_agent",
    description="An agent that builds resumes based on user requests.",
    instruction=(
        "You are a resume-writing assistant. Generate the entire resume yourself in markdown format "
        "with a clear title, section headers, and bullet points. Do not use any hardcoded template. "
        "When the markdown is ready, call save_pdf_to_local_folder with the full markdown text so it can be rendered to PDF. "
        "Only respond after the PDF has been saved."
    ),
    tools=[
        resume_builder_content_tool,
        convert_resume_to_md_tool,
        convert_md_to_pdf_tool,
        save_pdf_to_local_folder_tool,
    ],
    model="gemini-2.5-flash",
)

job_description_agent = Agent(
    name="job_description_agent",
    description="An agent that builds job descriptions based on user requests.",
    instruction=(
        "You are a job description-writing assistant. Take the brief produced by create_job_description and expand it into "
        "a complete, descriptive job description in markdown format with a clear title, section headers, and bullet points. "
        "Preserve the [JOB_DESCRIPTION] marker if present so the saver can route the file correctly. "
        "Do not copy the input verbatim or echo it back. Write the full job description content yourself. "
        "When the markdown is ready, call save_pdf_to_local_folder with the full markdown text so it can be rendered to PDF. "
        "Only respond after the PDF has been saved."
    ),
    tools=[
        create_job_description_tool,
        convert_job_description_to_md_tool,
        convert_md_to_pdf_tool,
        save_pdf_to_local_folder_tool,
    ],
    model="gemini-2.5-flash",
)
cordinator_agent = Agent(
    name="cordinator_agent",
    description="An agent that coordinates between the resume builder and job description agents.",
    instruction=(
        "You are a coordinator agent. Based on the user's request, determine whether to route the request to the "
        "resume builder agent or the job description agent. Use the appropriate tools to process the request and "
        "ensure that the final output is saved as a PDF. The save_pdf_to_local_folder tool is generic and can save either "
        "resume markdown or job description markdown."
    ),
    sub_agents=[resume_builder_agent, job_description_agent],
    model="gemini-2.5-flash",
)
agent_runner = InMemoryRunner(cordinator_agent)

if __name__ == "__main__":
    asyncio.run(
        agent_runner.session_service.create_session(
            app_name="InMemoryRunner",
            user_id="local-user",
            session_id="local-session",
        )
    )
    
    resume_details = """
Generate a professional ATS-friendly software engineer resume.

First Name: John
Last Name: Doe
Phone: +1 (555) 123-4567
Key Skills: Python, React, SQL, AWS, Docker
"""
    job_description_details = """
Generate a professional job description for a software engineer position.
Job Title: Software Engineer
Location: San Francisco, CA
Key Responsibilities:
- Develop and maintain web applications using Python and React."""

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=resume_details)],
    )
    for event in agent_runner.run(
        user_id="local-user",
        session_id="local-session",
        new_message=message,
    ):
        print(event)
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=job_description_details)],
    )
    for event in agent_runner.run(
        user_id="local-user",
        session_id="local-session",
        new_message=message,
    ):
        print(event)
