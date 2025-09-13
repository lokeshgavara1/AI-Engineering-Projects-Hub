import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="AI News Generator", page_icon="📰", layout="wide")

# Professional Header
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
    <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: bold;'>🤖 AI News Generator</h1>
    <p style='color: #e8e8e8; margin: 10px 0 0 0; font-size: 1.2rem;'>Powered by CrewAI and Cohere's Command R7B</p>
    <p style='color: #ffd700; margin: 5px 0 0 0; font-size: 1rem; font-style: italic;'>Generate comprehensive blog posts about any topic using advanced AI agents</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Content Settings")
    
    # Make the text input take up more space
    topic = st.text_area(
        "Enter your topic",
        height=100,
        placeholder="Enter the topic you want to generate content about..."
    )
    
    # Add more sidebar controls if needed
    st.markdown("### Advanced Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Add some spacing
    st.markdown("---")
    
    # Make the generate button more prominent in the sidebar
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)
    
    # Add some helpful information
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Enter your desired topic in the text area above
        2. Adjust the temperature if needed (higher = more creative)
        3. Click 'Generate Content' to start
        4. Wait for the AI to generate your article
        5. Download the result as a markdown file
        """)

    # Developer information in sidebar
    st.markdown("---")
    with st.container():
        st.markdown("### 👨‍💻 **Developer**")
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;'>
            <h4 style='margin: 0; color: #1f77b4;'>Lokesh Gavara</h4>
            <p style='margin: 5px 0; color: #666; font-style: italic;'>AI Engineer & Full Stack Developer</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lokeshgavara5/)")
        with col2:
            st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lokeshgavara1)")

    # Professional About Section
    with st.expander("📋 About the Developer"):
        st.markdown("""
        **Lokesh Gavara** is an experienced AI Engineer and Full Stack Developer specializing in:

        🔹 **Artificial Intelligence & Machine Learning**
        🔹 **CrewAI & Multi-Agent Systems**
        🔹 **Full Stack Web Development**
        🔹 **Python & Modern Frameworks**
        🔹 **AI-Powered Applications**

        This AI News Generator showcases advanced AI agent orchestration using CrewAI framework,
        demonstrating expertise in building intelligent, automated content generation systems.

        **Connect for collaborations, projects, or AI consulting opportunities!**
        """)

def generate_content(topic):
    # Check if API keys are set
    cohere_api_key = os.getenv("COHERE_API_KEY")
    serper_api_key = os.getenv("SERPER_API_KEY")

    if not cohere_api_key or cohere_api_key == "your_cohere_api_key_here":
        st.error("❌ Please set your COHERE_API_KEY in the .env file")
        return "Error: Missing Cohere API key"

    if not serper_api_key or serper_api_key == "your_serper_api_key_here":
        st.error("❌ Please set your SERPER_API_KEY in the .env file")
        return "Error: Missing Serper API key"

    llm = LLM(
        model="command-r",
        temperature=0.7,
        api_key=cohere_api_key
    )

    search_tool = SerperDevTool(
        n_results=10,
        api_key=serper_api_key
    )

    # First Agent: Senior Research Analyst
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",
        backstory="You're an expert research analyst with advanced web research skills. "
                "You excel at finding, analyzing, and synthesizing information from "
                "across the internet using search tools. You're skilled at "
                "distinguishing reliable sources from unreliable ones, "
                "fact-checking, cross-referencing information, and "
                "identifying key patterns and insights. You provide "
                "well-organized research briefs with proper citations "
                "and source verification. Your analysis includes both "
                "raw data and interpreted insights, making complex "
                "information accessible and actionable.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    # Second Agent: Content Writer
    content_writer = Agent(
        role="Content Writer",
        goal="Transform research findings into engaging blog posts while maintaining accuracy",
        backstory="You're a skilled content writer specialized in creating "
                "engaging, accessible content from technical research. "
                "You work closely with the Senior Research Analyst and excel at maintaining the perfect "
                "balance between informative and entertaining writing, "
                "while ensuring all facts and citations from the research "
                "are properly incorporated. You have a talent for making "
                "complex topics approachable without oversimplifying them.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Research Task
    research_task = Task(
        description=("""
            1. Conduct comprehensive research on {topic} including:
                - Recent developments and news
                - Key industry trends and innovations
                - Expert opinions and analyses
                - Statistical data and market insights
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a structured research brief
            4. Include all relevant citations and sources
        """),
        expected_output="""A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns
            Please format with clear sections and bullet points for easy reference.""",
        agent=senior_research_analyst
    )

    # Writing Task
    writing_task = Task(
        description=("""
            Using the research brief provided, create an engaging blog post that:
            1. Transforms technical information into accessible content
            2. Maintains all factual accuracy and citations from the research
            3. Includes:
                - Attention-grabbing introduction
                - Well-structured body sections with clear headings
                - Compelling conclusion
            4. Preserves all source citations in [Source: URL] format
            5. Includes a References section at the end
        """),
        expected_output="""A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes Inline citations hyperlinked to the original source url
            - Presents information in an accessible yet informative way
            - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections""",
        agent=content_writer
    )

    # Create Crew
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    return crew.kickoff(inputs={"topic": topic})

# Main content area
if not topic:
    st.info("👈 Please enter a topic in the sidebar to get started!")

    # Display setup instructions
    st.markdown("## 🚀 Getting Started")
    st.markdown("""
    ### 1. Get API Keys
    You'll need two API keys to use this application:

    - **[Serper API Key](https://serper.dev/)** - For web search functionality
    - **[Cohere API Key](https://dashboard.cohere.com/api-keys)** - For AI content generation

    ### 2. Configure Environment Variables
    Add your API keys to the `.env` file in the project directory:
    ```
    SERPER_API_KEY=your_serper_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    ```

    ### 3. Enter a Topic
    Use the sidebar to enter the topic you want to generate content about, then click "Generate Content".
    """)

elif generate_button:
    if not topic.strip():
        st.warning("Please enter a topic before generating content.")
    else:
        with st.spinner('Generating content... This may take a moment.'):
            try:
                result = generate_content(topic)

                # Check if result is an error message
                if isinstance(result, str) and result.startswith("Error:"):
                    st.error(result)
                else:
                    st.markdown("### Generated Content")
                    st.markdown(result)

                    # Add download button
                    st.download_button(
                        label="Download Content",
                        data=result.raw if hasattr(result, 'raw') else str(result),
                        file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                        mime="text/markdown"
                    )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.markdown("### Troubleshooting Tips:")
                st.markdown("""
                - Make sure your API keys are correctly set in the `.env` file
                - Check that you have internet connectivity
                - Try with a different topic
                - Restart the application if issues persist
                """)

# Professional Footer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin: 20px 0;'>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 300px;'>
            <h3 style='color: white; margin: 0 0 10px 0;'>🤖 AI News Generator</h3>
            <p style='color: #e8e8e8; margin: 0; font-size: 14px;'>
                Built with CrewAI, Streamlit and powered by Cohere's Command R7B<br>
                Leveraging advanced AI agents for comprehensive content generation
            </p>
        </div>
        <div style='flex: 0 0 auto; text-align: center; margin-left: 20px;'>
            <h4 style='color: white; margin: 0 0 15px 0;'>👨‍💻 Developed by</h4>
            <h3 style='color: #ffd700; margin: 0 0 10px 0;'>Lokesh Gavara</h3>
            <p style='color: #e8e8e8; margin: 0 0 15px 0; font-style: italic;'>AI Engineer & Full Stack Developer</p>
            <div style='display: flex; gap: 10px; justify-content: center;'>
                <a href='https://www.linkedin.com/in/lokeshgavara5/' target='_blank'>
                    <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='LinkedIn'/>
                </a>
                <a href='https://github.com/lokeshgavara1' target='_blank'>
                    <img src='https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white' alt='GitHub'/>
                </a>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Copyright notice
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    © 2024 Lokesh Gavara. All rights reserved. | Built with ❤️ using AI Technology
</div>
""", unsafe_allow_html=True)
