#!/usr/bin/env python3
"""
MCP Drive Server - Customer Support Document Operations
======================================================

A Model Context Protocol (MCP) server providing Google Drive, Docs, and Sheets integration
for customer support operations with integrated vector RAG capabilities for intelligent
FAQ and knowledge base search.

Key Features:
- Google Sheets integration for ticket logging and management
- Google Docs access for FAQ and knowledge base content
- Google Drive file operations and document management
- Vector RAG system with FAQ and KB search capabilities
- Automatic markdown conversion for AI-friendly document processing
- CSV formatting for structured sheet data
- Hybrid authentication (OAuth tokens + Service Account fallback)

MCP Tools Provided:
- log_ticket: Log customer support tickets to Google Sheets
- update_ticket_status: Update ticket status and add resolution notes
- search_tickets: Search existing tickets for patterns and history
- get_faq_content: Retrieve FAQ document content as markdown
- update_faq: Update FAQ document with new Q&A pairs
- initialize_faq_vectorstore: Create vector store from FAQ content
- search_faq_vectorized: Vector similarity search in FAQ database
- initialize_kb_vectorstore: Create vector store from PDF knowledge base
- search_kb_vectorized: Vector similarity search in knowledge base

MCP Resources Provided:
- sheet://tickets/active: Get active support tickets as CSV
- sheet://tickets/all: Get all tickets with full history
- doc://faq/all: Get complete FAQ document as markdown
- doc://ticket/{ticket_id}: Get specific ticket documentation

Environment Variables Required:
- PORT: Server port (default: 8002)
- DRIVE_TOKEN_JSON: OAuth token JSON string for persistent Drive access
- GOOGLE_SERVICE_ACCOUNT_JSON: Service account JSON for fallback authentication
- TICKETS_SHEET_ID: Google Sheets ID for customer support tickets
- FAQ_DOC_ID: Google Docs ID for FAQ document
- OPENAI_API_KEY: OpenAI API key for vector embeddings and LLM operations
- LANGSMITH_API_KEY: (Optional) LangSmith API key for tracing
- LANGSMITH_PROJECT: (Optional) LangSmith project name
- LANGSMITH_ENDPOINT: (Optional) LangSmith endpoint URL

Authentication Strategy:
1. Priority 1: Use OAuth token (DRIVE_TOKEN_JSON) for persistent consumer access
2. Priority 2: Fallback to Service Account (GOOGLE_SERVICE_ACCOUNT_JSON) for cloud deployment

Usage:
    python mcp_drive_server.py

Deployment:
    Render Cloud Platform (Port 8002)
    Requires Google Drive, Docs, and Sheets APIs enabled

Author: Multi-Agent Customer Support System
Version: 2.0.0
"""

# Standard library imports
import os
import json
import csv
import io
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Third-party imports
from pydantic import BaseModel, Field

# MCP SDK imports
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base

# Google API imports
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from dotenv import load_dotenv

load_dotenv()

# Local imports
from vector_rag_system import get_rag_system  # Vector RAG for FAQ/KB search

# =============================================================================
# MCP SERVER INITIALIZATION
# =============================================================================
# Initialize FastMCP server for Google Drive operations with vector RAG integration

mcp_drive = FastMCP("Google Drive Customer Support Server")

# =============================================================================
# GOOGLE DRIVE API CONFIGURATION
# =============================================================================
# Google Drive API scopes for comprehensive document and sheet operations

# Required scopes for customer support operations:
# - drive: Access to Drive files and folders
# - documents: Read/write Google Docs for FAQ and knowledge base
# - spreadsheets: Read/write Google Sheets for ticket management
DRIVE_SCOPES = [
    'https://www.googleapis.com/auth/drive',         # Drive file operations
    'https://www.googleapis.com/auth/documents',     # Google Docs access
    'https://www.googleapis.com/auth/spreadsheets'   # Google Sheets access
]

# =============================================================================
# ENVIRONMENT VARIABLE HELPERS
# =============================================================================
# Helper functions for environment variable management with clear error messages

def get_required_env_var(var_name: str, description: str) -> str:
    """
    Get required environment variable with clear error message.
    
    Args:
        var_name (str): Name of the environment variable
        description (str): Description of the variable for error messages
        
    Returns:
        str: Value of the environment variable
        
    Raises:
        ValueError: If the environment variable is not set
    """
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is required ({description})")
    return value

def get_env_var(var_name: str, default: str = None) -> str:
    """
    Get optional environment variable with default value.
    
    Args:
        var_name (str): Name of the environment variable
        default (str): Default value if variable is not set
        
    Returns:
        str: Value of the environment variable or default
    """
    return os.getenv(var_name, default)

# =============================================================================
# DATA MODELS
# =============================================================================
# Pydantic models for structured data exchange with AI agents

class TicketData(BaseModel):
    """
    Customer support ticket data model for Google Sheets integration.
    
    Represents a complete customer support ticket with all metadata required
    for tracking, assignment, and resolution. Used for logging new tickets
    and updating existing ticket status in Google Sheets.
    
    Attributes:
        ticket_id (str): Unique identifier for the ticket (auto-generated)
        customer_name (str): Full name of the customer submitting the ticket
        customer_email (str): Customer's email address for communication
        issue_type (str): Category of issue (technical, billing, general, account)
        priority (str): Priority level (low, medium, high, urgent)
        description (str): Detailed description of the customer's issue
        status (str): Current ticket status (open, in_progress, resolved, closed)
        assigned_agent (str): Name/ID of the support agent handling the ticket
        created_at (str): ISO timestamp when ticket was created
        updated_at (str): ISO timestamp of last update
        resolution (Optional[str]): Detailed resolution notes (when resolved)
        
    Example:
        ticket = TicketData(
            ticket_id="TKT-2024-001",
            customer_name="John Doe",
            customer_email="john@example.com",
            issue_type="technical",
            priority="high",
            description="Cannot access dashboard after login",
            status="open",
            assigned_agent="Agent Smith",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:00:00Z"
        )
    """
    ticket_id: str = Field(description="Unique ticket identifier")
    customer_name: str = Field(description="Customer name")
    customer_email: str = Field(description="Customer email address")
    issue_type: str = Field(description="Type of issue (technical, billing, general)")
    priority: str = Field(description="Priority level (low, medium, high, urgent)")
    description: str = Field(description="Issue description")
    status: str = Field(description="Current status")
    assigned_agent: str = Field(description="Assigned support agent")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Last update timestamp")
    resolution: Optional[str] = Field(description="Resolution details", default=None)

class SheetUpdateResult(BaseModel):
    """
    Google Sheets update operation result model.
    
    Represents the result of a Google Sheets operation such as adding rows,
    updating cell values, or modifying sheet structure.
    
    Attributes:
        status (str): Operation status (success, error, partial)
        sheet_id (str): Google Sheets document ID
        range_updated (str): A1 notation of the updated range (e.g., "A1:D10")
        rows_added (int): Number of new rows added to the sheet
        updated_at (str): ISO timestamp of the operation
    """
    status: str = Field(description="Update status")
    sheet_id: str = Field(description="Google Sheet ID")
    range_updated: str = Field(description="Updated range")
    rows_added: int = Field(description="Number of rows added")
    updated_at: str = Field(description="Update timestamp")

class DocumentResult(BaseModel):
    """
    Google Docs operation result model.
    
    Represents the result of a Google Docs operation such as reading content,
    updating text, or retrieving document metadata.
    
    Attributes:
        status (str): Operation status (success, error)
        document_id (str): Google Docs document ID
        title (str): Document title
        content_markdown (str): Document content converted to markdown for AI processing
        word_count (int): Total word count in the document
        updated_at (str): ISO timestamp of last document modification
    """
    status: str = Field(description="Operation status")
    document_id: str = Field(description="Google Doc ID")
    title: str = Field(description="Document title")
    content_markdown: str = Field(description="Document content in markdown")
    word_count: int = Field(description="Word count")
    updated_at: str = Field(description="Last update timestamp")

# =============================================================================
# GOOGLE DRIVE SERVICE CLASS
# =============================================================================
# Core service class for Google Drive, Docs, and Sheets operations

class DriveService:
    """
    Google Drive service class for customer support document operations.
    
    Handles authentication and provides unified access to Google Drive, Docs, and Sheets APIs.
    Supports hybrid authentication with OAuth tokens and Service Account fallback for
    flexible deployment scenarios.
    
    Features:
    - Hybrid authentication (OAuth + Service Account)
    - Google Sheets integration for ticket management
    - Google Docs access for FAQ and knowledge base
    - Google Drive file operations
    - Automatic markdown conversion for AI processing
    - CSV formatting for structured sheet data
    
    Attributes:
        drive_service: Authenticated Google Drive API service
        docs_service: Authenticated Google Docs API service
        sheets_service: Authenticated Google Sheets API service
        SCOPES: Required Google API scopes
    """
    
    def __init__(self):
        """
        Initialize Google Drive service with authentication.
        
        Sets up authenticated connections to Google Drive, Docs, and Sheets APIs
        using hybrid authentication strategy (OAuth token with Service Account fallback).
        
        Raises:
            ValueError: If authentication fails or required credentials are missing
        """
        # Initialize service instances
        self.drive_service = None
        self.docs_service = None
        self.sheets_service = None
        self.SCOPES = DRIVE_SCOPES
        
        # Authenticate and initialize all services
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google APIs using hybrid authentication"""
        try:
            # Priority 1: Try OAuth token JSON (for persistent consumer Drive access)
            drive_token_json = os.getenv('DRIVE_TOKEN_JSON')
            if drive_token_json:
                print("🔑 Using OAuth token from DRIVE_TOKEN_JSON")
                
                try:
                    # Parse token JSON
                    token_data = json.loads(drive_token_json)
                    creds = Credentials.from_authorized_user_info(token_data, self.SCOPES)
                    
                    # Refresh if expired
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                        print("🔄 OAuth token refreshed")
                        # Note: In production, refreshed token would need to be saved back to env var
                        # For now, we'll use the refreshed token for this session
                    
                    if creds and creds.valid:
                        self.drive_service = build('drive', 'v3', credentials=creds)
                        self.docs_service = build('docs', 'v1', credentials=creds)
                        self.sheets_service = build('sheets', 'v4', credentials=creds)
                        print("✅ Drive service authenticated with OAuth token")
                        return
                    else:
                        print("⚠️  OAuth token invalid, falling back to service account")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  Invalid DRIVE_TOKEN_JSON format: {e}, falling back to service account")
            
            # Priority 2: Fallback to service account (for non-consumer scenarios)
            credentials_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not credentials_json:
                raise ValueError("Neither DRIVE_TOKEN_JSON nor GOOGLE_SERVICE_ACCOUNT_JSON environment variable set")
            
            print("🔑 Using service account authentication")
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=self.SCOPES
            )
            
            self.drive_service = build('drive', 'v3', credentials=credentials)
            self.docs_service = build('docs', 'v1', credentials=credentials)
            self.sheets_service = build('sheets', 'v4', credentials=credentials)
            print("✅ Drive service authenticated with service account")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
        except Exception as e:
            raise Exception(f"Google Drive authentication failed: {e}")
    
    def _doc_to_markdown(self, doc_content) -> str:
        """Convert Google Doc content to markdown format"""
        markdown = ""
        
        for element in doc_content.get('body', {}).get('content', []):
            if 'paragraph' in element:
                paragraph_text = ""
                for text_run in element['paragraph'].get('elements', []):
                    if 'textRun' in text_run:
                        text = text_run['textRun'].get('content', '')
                        text_style = text_run['textRun'].get('textStyle', {})
                        
                        # Apply markdown formatting
                        if text_style.get('bold'):
                            text = f"**{text}**"
                        if text_style.get('italic'):
                            text = f"*{text}*"
                        if text_style.get('underline'):
                            text = f"_{text}_"
                        
                        paragraph_text += text
                
                # Handle paragraph styles
                paragraph_style = element['paragraph'].get('paragraphStyle', {})
                named_style = paragraph_style.get('namedStyleType', '')
                
                if named_style == 'HEADING_1':
                    markdown += f"# {paragraph_text}"
                elif named_style == 'HEADING_2':
                    markdown += f"## {paragraph_text}"
                elif named_style == 'HEADING_3':
                    markdown += f"### {paragraph_text}"
                else:
                    markdown += paragraph_text
        
        return markdown.strip()
    
    def _sheet_to_csv(self, values) -> str:
        """Convert sheet values to CSV format"""
        if not values:
            return ""
        
        output = io.StringIO()
        writer = csv.writer(output)
        for row in values:
            writer.writerow(row)
        return output.getvalue()

# Initialize Drive service
drive_service = DriveService()

# MCP Tools
@mcp_drive.tool()
async def log_ticket_to_sheet(ticket_data: TicketData) -> SheetUpdateResult:
    """Log customer support ticket to Google Sheets"""
    try:
        # Get tickets sheet ID from environment variable
        TICKETS_SHEET_ID = get_required_env_var('TICKETS_SHEET_ID', 'Google Sheets ID for customer support tickets')
        
        # Prepare row data
        row_data = [
            ticket_data.ticket_id,
            ticket_data.customer_name,
            ticket_data.customer_email,
            ticket_data.issue_type,
            ticket_data.priority,
            ticket_data.description,
            ticket_data.status,
            ticket_data.assigned_agent,
            ticket_data.created_at,
            ticket_data.updated_at,
            ticket_data.resolution or ""
        ]
        
        # Append to sheet
        result = drive_service.sheets_service.spreadsheets().values().append(
            spreadsheetId=TICKETS_SHEET_ID,
            range="Tickets!A:K",
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body={'values': [row_data]}
        ).execute()
        
        return SheetUpdateResult(
            status="success",
            sheet_id=TICKETS_SHEET_ID,
            range_updated=result.get('updates', {}).get('updatedRange', ''),
            rows_added=result.get('updates', {}).get('updatedRows', 0),
            updated_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        return SheetUpdateResult(
            status="error",
            sheet_id="",
            range_updated="",
            rows_added=0,
            updated_at=datetime.now().isoformat()
        )

@mcp_drive.tool()
async def update_ticket_status(ticket_id: str, status: str, notes: str = "") -> SheetUpdateResult:
    """Update ticket status in Google Sheets"""
    try:
        TICKETS_SHEET_ID = get_required_env_var('TICKETS_SHEET_ID', 'Google Sheets ID for customer support tickets')
        
        # Find the ticket row
        result = drive_service.sheets_service.spreadsheets().values().get(
            spreadsheetId=TICKETS_SHEET_ID,
            range="Tickets!A:K"
        ).execute()
        
        values = result.get('values', [])
        row_to_update = None
        
        for i, row in enumerate(values):
            if len(row) > 0 and row[0] == ticket_id:
                row_to_update = i + 1  # Sheets are 1-indexed
                break
        
        if row_to_update:
            # Update status and timestamp
            update_data = [
                [status, datetime.now().isoformat(), notes]
            ]
            
            drive_service.sheets_service.spreadsheets().values().update(
                spreadsheetId=TICKETS_SHEET_ID,
                range=f"Tickets!G{row_to_update}:I{row_to_update}",
                valueInputOption='RAW',
                body={'values': update_data}
            ).execute()
            
            return SheetUpdateResult(
                status="success",
                sheet_id=TICKETS_SHEET_ID,
                range_updated=f"G{row_to_update}:I{row_to_update}",
                rows_added=0,
                updated_at=datetime.now().isoformat()
            )
        else:
            return SheetUpdateResult(
                status="error",
                sheet_id=TICKETS_SHEET_ID,
                range_updated="",
                rows_added=0,
                updated_at=datetime.now().isoformat()
            )
    
    except Exception as e:
        return SheetUpdateResult(
            status="error",
            sheet_id="",
            range_updated="",
            rows_added=0,
            updated_at=datetime.now().isoformat()
        )

@mcp_drive.tool()
async def update_faq_doc(topic: str, content: str) -> DocumentResult:
    """Update FAQ document with new content"""
    try:
        # Default FAQ doc ID (should be configured)
        FAQ_DOC_ID = get_required_env_var('FAQ_DOC_ID', 'Google Docs ID for FAQ document')
        
        # Get current document
        doc = drive_service.docs_service.documents().get(documentId=FAQ_DOC_ID).execute()
        content_end_index = doc['body']['content'][-1]['endIndex'] - 1
        
        # Format content as FAQ entry
        faq_entry = f"\n\n## {topic}\n\n{content}\n\n---"
        
        # Insert content at the end
        requests = [{
            'insertText': {
                'location': {'index': content_end_index},
                'text': faq_entry
            }
        }]
        
        # Execute the update
        drive_service.docs_service.documents().batchUpdate(
            documentId=FAQ_DOC_ID,
            body={'requests': requests}
        ).execute()
        
        # Get updated document for response
        updated_doc = drive_service.docs_service.documents().get(documentId=FAQ_DOC_ID).execute()
        content_markdown = drive_service._doc_to_markdown(updated_doc)
        
        return DocumentResult(
            status="success",
            document_id=FAQ_DOC_ID,
            title=updated_doc.get('title', 'FAQ Document'),
            content_markdown=content_markdown,
            word_count=len(content_markdown.split()),
            updated_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        return DocumentResult(
            status="error",
            document_id="",
            title="",
            content_markdown="",
            word_count=0,
            updated_at=datetime.now().isoformat()
        )

@mcp_drive.tool()
async def create_ticket_doc(ticket_id: str, details: str) -> DocumentResult:
    """Create detailed ticket documentation"""
    try:
        # Create new document
        doc_title = f"Support Ticket {ticket_id} - Details"
        
        create_request = {
            'title': doc_title
        }
        
        doc = drive_service.docs_service.documents().create(body=create_request).execute()
        doc_id = doc['documentId']
        
        # Add content to document
        content = f"""# Support Ticket: {ticket_id}

## Ticket Details

{details}

## Resolution Log

*This section will be updated as the ticket progresses.*

---

Created: {datetime.now().isoformat()}
"""
        
        requests = [{
            'insertText': {
                'location': {'index': 1},
                'text': content
            }
        }]
        
        drive_service.docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': requests}
        ).execute()
        
        return DocumentResult(
            status="success",
            document_id=doc_id,
            title=doc_title,
            content_markdown=content,
            word_count=len(content.split()),
            updated_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        return DocumentResult(
            status="error",
            document_id="",
            title="",
            content_markdown="",
            word_count=0,
            updated_at=datetime.now().isoformat()
        )

@mcp_drive.tool()
async def search_knowledge_base(query: str) -> List[DocumentResult]:
    """Search existing documentation for relevant information"""
    try:
        # Search for documents containing the query
        search_query = f"fullText contains '{query}'"
        
        results = drive_service.drive_service.files().list(
            q=search_query,
            fields="files(id, name, mimeType, modifiedTime)",
            pageSize=10
        ).execute()
        
        documents = []
        
        for file in results.get('files', []):
            if file['mimeType'] == 'application/vnd.google-apps.document':
                try:
                    doc = drive_service.docs_service.documents().get(documentId=file['id']).execute()
                    content_markdown = drive_service._doc_to_markdown(doc)
                    
                    documents.append(DocumentResult(
                        status="success",
                        document_id=file['id'],
                        title=file['name'],
                        content_markdown=content_markdown,
                        word_count=len(content_markdown.split()),
                        updated_at=file.get('modifiedTime', datetime.now().isoformat())
                    ))
                except:
                    continue
        
        return documents
    
    except Exception as e:
        return []

# MCP Resources
@mcp_drive.resource("sheet://tickets/active")
def get_active_tickets() -> str:
    """Get active tickets as CSV data"""
    try:
        TICKETS_SHEET_ID = get_required_env_var('TICKETS_SHEET_ID', 'Google Sheets ID for customer support tickets')
        
        result = drive_service.sheets_service.spreadsheets().values().get(
            spreadsheetId=TICKETS_SHEET_ID,
            range="Tickets!A:K"
        ).execute()
        
        values = result.get('values', [])
        
        # Filter for active tickets (status not 'resolved' or 'closed')
        active_tickets = []
        if values:
            headers = values[0] if values else []
            active_tickets.append(headers)  # Add headers
            
            for row in values[1:]:
                if len(row) > 6 and row[6].lower() not in ['resolved', 'closed']:
                    active_tickets.append(row)
        
        return drive_service._sheet_to_csv(active_tickets)
    
    except Exception as e:
        return f"Error retrieving active tickets: {str(e)}"

@mcp_drive.resource("sheet://tickets/resolved")
def get_resolved_tickets() -> str:
    """Get resolved tickets for analysis (CSV format)"""
    try:
        TICKETS_SHEET_ID = get_required_env_var('TICKETS_SHEET_ID', 'Google Sheets ID for customer support tickets')
        
        result = drive_service.sheets_service.spreadsheets().values().get(
            spreadsheetId=TICKETS_SHEET_ID,
            range="Tickets!A:K"
        ).execute()
        
        values = result.get('values', [])
        
        # Filter for resolved tickets
        resolved_tickets = []
        if values:
            headers = values[0] if values else []
            resolved_tickets.append(headers)
            
            for row in values[1:]:
                if len(row) > 6 and row[6].lower() in ['resolved', 'closed']:
                    resolved_tickets.append(row)
        
        return drive_service._sheet_to_csv(resolved_tickets)
    
    except Exception as e:
        return f"Error retrieving resolved tickets: {str(e)}"

@mcp_drive.resource("doc://faq/{topic}")
def get_faq_by_topic(topic: str) -> str:
    """Get FAQ document content as markdown"""
    try:
        FAQ_DOC_ID = get_required_env_var('FAQ_DOC_ID', 'Google Docs ID for FAQ document')
        
        doc = drive_service.docs_service.documents().get(documentId=FAQ_DOC_ID).execute()
        content_markdown = drive_service._doc_to_markdown(doc)
        
        # Filter content by topic if specified
        if topic.lower() != 'all':
            lines = content_markdown.split('\n')
            topic_content = []
            in_topic_section = False
            
            for line in lines:
                if line.startswith('##') and topic.lower() in line.lower():
                    in_topic_section = True
                    topic_content.append(line)
                elif line.startswith('##') and in_topic_section:
                    break
                elif in_topic_section:
                    topic_content.append(line)
            
            return '\n'.join(topic_content) if topic_content else f"No FAQ found for topic: {topic}"
        
        return content_markdown
    
    except Exception as e:
        return f"Error retrieving FAQ for topic {topic}: {str(e)}"

@mcp_drive.resource("doc://ticket/{ticket_id}")
def get_ticket_doc(ticket_id: str) -> str:
    """Get ticket documentation as markdown"""
    try:
        # Search for ticket document
        search_query = f"name contains 'Support Ticket {ticket_id}'"
        
        results = drive_service.drive_service.files().list(
            q=search_query,
            fields="files(id, name)",
            pageSize=1
        ).execute()
        
        files = results.get('files', [])
        if not files:
            return f"No documentation found for ticket {ticket_id}"
        
        doc_id = files[0]['id']
        doc = drive_service.docs_service.documents().get(documentId=doc_id).execute()
        content_markdown = drive_service._doc_to_markdown(doc)
        
        return content_markdown
    
    except Exception as e:
        return f"Error retrieving ticket documentation for {ticket_id}: {str(e)}"

# MCP Prompts
@mcp_drive.prompt(title="Ticket Logging Template")
def ticket_logging_template(customer_name: str, issue_type: str, priority: str) -> List[base.Message]:
    """Generate structured ticket logging format"""
    return [
        base.UserMessage(f"Create a ticket log entry for customer {customer_name}"),
        base.UserMessage(f"Issue type: {issue_type}"),
        base.UserMessage(f"Priority: {priority}"),
        base.AssistantMessage("I'll create a structured ticket entry with all necessary fields for tracking and resolution.")
    ]

@mcp_drive.prompt(title="FAQ Update Template")
def faq_update_template(topic: str, question: str, answer: str) -> List[base.Message]:
    """Generate FAQ content update guidelines"""
    return [
        base.UserMessage(f"Update FAQ for topic: {topic}"),
        base.UserMessage(f"Question: {question}"),
        base.UserMessage(f"Answer: {answer}"),
        base.AssistantMessage("I'll format this as a clear, searchable FAQ entry with proper markdown structure.")
    ]

@mcp_drive.prompt(title="Resolution Documentation Template")
def resolution_documentation_template(ticket_id: str, issue_summary: str, solution: str) -> List[base.Message]:
    """Generate solution documentation format"""
    return [
        base.UserMessage(f"Document resolution for ticket {ticket_id}"),
        base.UserMessage(f"Issue summary: {issue_summary}"),
        base.UserMessage(f"Solution applied: {solution}"),
        base.AssistantMessage("I'll create comprehensive documentation that can be referenced for similar future issues.")
    ]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8002))
    uvicorn.run(mcp_drive, host="0.0.0.0", port=port)