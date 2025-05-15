from app import db
from datetime import datetime
from sqlalchemy.orm import relationship

class Project(db.Model):
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    test_runs = relationship("TestRun", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project {self.name}>"

class TestRun(db.Model):
    __tablename__ = 'test_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    test_type = db.Column(db.String(50), nullable=False)  # model-based, mutation, reinforcement, behavior-driven, fuzz
    input_type = db.Column(db.String(50), nullable=False)  # requirements, code
    input_content = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="pending")  # pending, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.now)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="test_runs")
    test_cases = relationship("TestCase", back_populates="test_run", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TestRun {self.id} - {self.test_type}>"

class TestCase(db.Model):
    __tablename__ = 'test_cases'
    
    id = db.Column(db.Integer, primary_key=True)
    test_run_id = db.Column(db.Integer, db.ForeignKey('test_runs.id', ondelete='CASCADE'), nullable=False)
    test_id = db.Column(db.String(20), nullable=False)  # e.g., TC-1
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    preconditions = db.Column(db.Text, nullable=True)  # Keep the original column name for now
    expected_result = db.Column(db.Text, nullable=False)
    steps = db.Column(db.JSON, nullable=False)  # List of steps as JSON
    
    # Relationships
    test_run = relationship("TestRun", back_populates="test_cases")
    
    def __repr__(self):
        return f"<TestCase {self.test_id} - {self.name}>"

class UserSettings(db.Model):
    __tablename__ = 'user_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    theme = db.Column(db.String(50), default="dark")  # dark, light, contrast
    background_color = db.Column(db.String(50), default="#1e1e1e")  # CSS color
    text_color = db.Column(db.String(50), default="#ffffff")  # CSS color
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<UserSettings {self.id}>"
