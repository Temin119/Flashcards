import React, { useState, useEffect, useRef } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { 
  FileDown, 
  MessageSquare, 
  BookOpen, 
  PenLine,
  Pencil, 
  Brain, 
  HelpCircle, 
  ChevronLeft, 
  ChevronRight, 
  Volume2,
  PanelLeftClose,
  PanelLeftOpen,
  ArrowLeft,
  ArrowRight,
  ExternalLink,
  AlertCircle,
  Shuffle
} from 'lucide-react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import ReactMarkdown from 'react-markdown';
import type { ChatMessage, Document } from '../lib/firestore';
import PdfViewer from './PdfViewer';
import YoutubeViewer from './YoutubeViewer';
import LectureTranscript from './LectureTranscript';
import Chat from './Chat';
import LoadingBook from './LoadingBook';
import LoadingBrain from './LoadingBrain';
import LoadingTest from './LoadingTest';
import LoadingNotes from './LoadingNotes';
import LoadingAudio from './LoadingAudio';
import Flashcard from './Flashcard';
import AudioCard from './AudioCard';
import { generateSummary, generateFlashcards, generateTest, generateNotes, generateStyledText, type TestQuestion } from '../lib/openai';
import { updateDocument, updateDocumentTitle } from '../lib/firestore';
import { generateSpeech, type AudioStyle } from '../lib/elevenlabs';
import { fetchDocumentAudios, deleteAudio, renameAudio, MAX_AUDIOS_PER_DOCUMENT } from '../lib/storage';
import VocalizeTab from './VocalizeTab';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

type Tab = 'chat' | 'notes' | 'flashcards' | 'test' | 'vocalize';

interface DocumentViewProps {
  documents: Document[];
  onDocumentsChange?: () => Promise<void>;
}

interface TestAnswer {
  questionIndex: number;
  selectedAnswer: string;
}

interface AudioEntry {
  url: string;
  style: AudioStyle;
  fileName?: string;
}

export default function DocumentView({ documents, onDocumentsChange }: DocumentViewProps) {
  const { id } = useParams<{ id: string }>();
  const document = documents.find(doc => doc.id === id);
  const userId = localStorage.getItem('userId');

  const [activeTab, setActiveTab] = useState<Tab>('chat');
  const [scale, setScale] = useState(1.0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentFlashcardIndex, setCurrentFlashcardIndex] = useState(0);
  const [slideDirection, setSlideDirection] = useState<'left' | 'right' | undefined>();
  const [localFlashcards, setLocalFlashcards] = useState(document?.flashcards || []);
  const [localNotes, setLocalNotes] = useState(document?.notes || '');
  const [localTest, setLocalTest] = useState<TestQuestion[]>(document?.test || []);
  const [isHovered, setIsHovered] = useState(false);
  const [testAnswers, setTestAnswers] = useState<TestAnswer[]>([]);
  const [testSubmitted, setTestSubmitted] = useState(false);
  const [testScore, setTestScore] = useState<number>(0);
  const [audioStyle, setAudioStyle] = useState<AudioStyle>(document?.audioStyle || 'Lecture');
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isPdfVisible, setIsPdfVisible] = useState(true);
  const tabsContainerRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(document?.audioUrl || null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [youtubePlayerRef, setYoutubePlayerRef] = useState<any>(null);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [newTitle, setNewTitle] = useState(document?.title || '');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(document?.chatHistory || []);
  const [savedAudios, setSavedAudios] = useState<AudioEntry[]>([]);
  const [currentPlayingUrl, setCurrentPlayingUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const titleInputRef = useRef<HTMLInputElement>(null);
  const [startPage, setStartPage] = useState<number>();
  const [endPage, setEndPage] = useState<number>();
  const [specificPages, setSpecificPages] = useState<string>('');

  const handleTimestampClick = (time: number) => {
    if (youtubePlayerRef) {
      youtubePlayerRef.seekTo(time);
      youtubePlayerRef.playVideo();
    }
  };

  // Check if we're on mobile
  const isMobile = window.innerWidth <= 768;

  // Update scroll buttons visibility
  const updateScrollButtons = () => {
    if (tabsContainerRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = tabsContainerRef.current;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft < scrollWidth - clientWidth);
    }
  };

  // Add scroll event listener
  useEffect(() => {
    const tabsContainer = tabsContainerRef.current;
    if (tabsContainer) {
      tabsContainer.addEventListener('scroll', updateScrollButtons);
      updateScrollButtons();
      return () => tabsContainer.removeEventListener('scroll', updateScrollButtons);
    }
  }, []);

  // Add resize event listener
  useEffect(() => {
    const handleResize = () => {
      updateScrollButtons();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const scrollTabs = (direction: 'left' | 'right') => {
    if (tabsContainerRef.current) {
      const scrollAmount = 200;
      tabsContainerRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth'
      });
    }
  };

  // Update local states when document changes
  useEffect(() => {
    const loadDocumentData = async () => {
      if (document && userId) {
        // Only update document-related states
        setLocalFlashcards(document.flashcards || []);
        setLocalNotes(document.notes || '');
        setLocalTest(document.test || []);
        setNewTitle(document.title);
        setAudioStyle(document.audioStyle || 'Lecture');

        // Fetch all audio files for this document
        const audios = await fetchDocumentAudios(userId, document.id);
        setSavedAudios(audios);
      }
    };
  
    loadDocumentData();
  }, [document?.id, userId]);

  const handleChatUpdate = async (messages: ChatMessage[]) => {
    setChatMessages(messages);
    if (document) {
      try {
        await updateDocument(document.id, { chatHistory: messages });
        if (onDocumentsChange) {
          await onDocumentsChange();
        }
      } catch (error) {
        console.error('Error updating chat history:', error);
      }
    }
  };

  const handleTranscriptUpdate = async (newContent: string) => {
    if (!document) return;

    try {
      await updateDocument(document.id, { content: newContent });
      if (onDocumentsChange) {
        await onDocumentsChange();
      }
    } catch (error) {
      console.error('Error updating transcript:', error);
      setError('Failed to update transcript');
    }
  };

  // Clean up audio URL when component unmounts
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, []);

  const handleContentUpdate = async (content: string) => {
    if (!document?.id) return;
    try {
      await updateDocument(document.id, { content });
      if (onDocumentsChange) {
        await onDocumentsChange();
      }
    } catch (error) {
      console.error('Error updating document content:', error);
      setError('Failed to update document content');
    }
  };

  if (!document) {
    return <Navigate to="/dashboard" replace />;
  }

  const tabs: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'notes', label: 'Notes', icon: PenLine },
    { id: 'flashcards', label: 'Flashcards', icon: Brain },
    { id: 'test', label: 'Test', icon: HelpCircle },
    { id: 'vocalize' as const, label: 'Vocalize', icon: Volume2, beta: true }
  ];

  const handleGenerate = async () => {
    if (!document.content || isGenerating) return;
    setIsGenerating(true);
    setError(null);

    try {
      if (activeTab === 'notes') {
        const notes = await generateNotes(document.content);
        await updateDocument(document.id, { notes });
        setLocalNotes(notes);
      } else if (activeTab === 'flashcards') {
        // Parse and validate page numbers
        const parsedStartPage = startPage ? parseInt(startPage.toString()) : undefined;
        const parsedEndPage = endPage ? parseInt(endPage.toString()) : undefined;
        const parsedSpecificPages = specificPages.trim() 
          ? specificPages.split(',')
              .map(p => parseInt(p.trim()))
              .filter(p => !isNaN(p) && p > 0 && p <= document.numPages)
          : undefined;

        // Validate page ranges
        if (parsedStartPage && parsedStartPage > document.numPages) {
          setError(`Invalid start page. This document only has ${document.numPages} pages.`);
          setIsGenerating(false);
          return;
        }
        if (parsedEndPage && parsedEndPage > document.numPages) {
          setError(`Invalid end page. This document only has ${document.numPages} pages.`);
          setIsGenerating(false);
          return;
        }
        if (parsedStartPage && parsedEndPage && parsedStartPage > parsedEndPage) {
          setError('Start page cannot be greater than end page.');
          setIsGenerating(false);
          return;
        }

        // Generate flashcards with validated page numbers
        const flashcards = await generateFlashcards(
          document.content, 
          parsedStartPage, 
          parsedEndPage, 
          parsedSpecificPages
        );
        
        await updateDocument(document.id, { flashcards });
        setLocalFlashcards(flashcards);
        setCurrentFlashcardIndex(0);
      } else if (activeTab === 'test') {
        const questions = await generateTest(document.content);
        await updateDocument(document.id, { test: questions });
        setLocalTest(questions);
        setTestAnswers([]);
        setTestSubmitted(false);
        setTestScore(0);
      }
    } catch (error) {
      console.error(`Error generating ${activeTab}:`, error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePrevCard = () => {
    setSlideDirection('right');
    setCurrentFlashcardIndex((prev) => 
      prev === 0 ? localFlashcards.length - 1 : prev - 1
    );
  };

  const handleNextCard = () => {
    setSlideDirection('left');
    setCurrentFlashcardIndex((prev) => 
      prev === localFlashcards.length - 1 ? 0 : prev + 1
    );
  };

  const handleAnswerSelect = (questionIndex: number, answer: string) => {
    setTestAnswers(prev => {
      const newAnswers = [...prev];
      const existingIndex = newAnswers.findIndex(a => a.questionIndex === questionIndex);
      
      if (existingIndex !== -1) {
        newAnswers[existingIndex].selectedAnswer = answer;
      } else {
        newAnswers.push({ questionIndex, selectedAnswer: answer });
      }
      
      return newAnswers;
    });
  };

  const handleTestSubmit = () => {
    if (testAnswers.length < localTest.length) {
      alert('Please answer all questions before submitting.');
      return;
    }

    const correctAnswers = testAnswers.reduce((count, answer) => {
      const question = localTest[answer.questionIndex];
      return count + (answer.selectedAnswer === question.correctAnswer ? 1 : 0);
    }, 0);

    const percentage = Math.round((correctAnswers / localTest.length) * 100);
    setTestScore(percentage);
    setTestSubmitted(true);
  };

  const handleGenerateAudio = async () => {
    if (!document?.content || isGeneratingAudio || !userId) {
      if (!userId) {
        alert('Please log in to generate audio');
      }
      return;
    }

    try {
      setIsGeneratingAudio(true);

      // Generate styled text using OpenAI
      const styledText = await generateStyledText(document.content, audioStyle);

      // Convert the styled text to speech using Eleven Labs
      const newAudioUrl = await generateSpeech(styledText, audioStyle, document.id, userId);
      
      // Create new audio entry
      const newAudio: AudioEntry = {
        url: newAudioUrl,
        style: audioStyle
      };

      // Update saved audios
      const updatedAudios = [...savedAudios, newAudio];
      setSavedAudios(updatedAudios);

      // Update document with new audio
      await updateDocument(document.id, {
        audioUrl: newAudioUrl,
        audioStyle,
        additionalAudios: updatedAudios.slice(1) // Store additional audios separately
      });

      setIsGeneratingAudio(false);

      // Auto-play the new audio
      setCurrentPlayingUrl(newAudioUrl);
      if (audioRef.current) {
        try {
          await audioRef.current.play();
          setIsAudioPlaying(true);
        } catch (playError) {
          console.warn('Auto-play failed:', playError);
        }
      }
    } catch (error) {
      console.error('Error generating audio:', error);
      alert(error instanceof Error ? error.message : 'Failed to generate audio. Please try again.');
      setIsGeneratingAudio(false);
    }
  };

  const handleTitleUpdate = async () => {
    if (!newTitle.trim() || newTitle === document.title) {
      setIsEditingTitle(false);
      setNewTitle(document.title);
      return;
    }
  
    try {
      const result = await updateDocumentTitle(document.id, newTitle.trim());
      if (result.success) {
        await onDocumentsChange?.();
        setIsEditingTitle(false);
      } else {
        setNewTitle(document.title);
        setIsEditingTitle(false);
      }
    } catch (error) {
      console.error('Failed to update title:', error);
      setNewTitle(document.title);
      setIsEditingTitle(false);
    }
  };

  useEffect(() => {
    if (isEditingTitle && titleInputRef.current) {
      titleInputRef.current.focus();
      titleInputRef.current.select();
    }
  }, [isEditingTitle]);

  // Clean up audio URL when component unmounts or audio style changes
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const handleShuffle = () => {
    const randomIndex = Math.floor(Math.random() * localFlashcards.length);
    setSlideDirection('left');
    setCurrentFlashcardIndex(randomIndex);
  };

  const renderTabContent = () => {
    switch (activeTab) {

      case 'chat':
        return (
          <Chat 
            documentContent={document.content} 
            documentId={document.id}
            initialMessages={chatMessages}
            onChatUpdate={handleChatUpdate}
            selectedText={selectedText}
            selectedImage={selectedImage}
            onClearSelection={() => {
              setSelectedText(null);
              setSelectedImage(null);
            }}
            documentType={document.type}
            youtubeUrl={document.youtube_link}
            onTimestampClick={handleTimestampClick}
          />
        );

        case 'notes':
          if (localNotes) {
            return (
              <div className="prose prose-indigo max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    // Headers with gradients and proper spacing
                    h1: ({ children }) => (
                      <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent pb-4 mb-8 border-b-2 border-indigo-100">
                        {children}
                      </h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-2xl font-semibold text-gray-800 mt-12 mb-6 pb-2 border-b border-indigo-50">
                        {children}
                      </h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-xl font-medium text-gray-700 mt-8 mb-4">
                        {children}
                      </h3>
                    ),
                    // Paragraphs with improved readability
                    p: ({ children }) => (
                      <p className="text-gray-600 leading-relaxed mb-6 text-lg">
                        {children}
                      </p>
                    ),
                    // Lists with custom bullets and spacing
                    ul: ({ children }) => (
                      <ul className="space-y-3 my-6 ml-4">
                        {children}
                      </ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="space-y-3 my-6 ml-4 list-decimal">
                        {children}
                      </ol>
                    ),
                    li: ({ children }) => (
                      <li className="flex items-start space-x-3">
                        <span className="w-2 h-2 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full mt-2.5 flex-shrink-0" />
                        <span className="text-gray-600">{children}</span>
                      </li>
                    ),
                    // Blockquotes with gradient border and background
                    blockquote: ({ children }) => (
                      <blockquote className="my-8 pl-6 border-l-4 border-gradient-to-b from-indigo-500 to-purple-500 bg-gradient-to-r from-indigo-50 to-purple-50 py-4 px-6 rounded-r-lg">
                        <p className="text-gray-700 italic">{children}</p>
                      </blockquote>
                    ),
                    // Section dividers
                    hr: () => (
                      <hr className="my-12 border-none h-px bg-gradient-to-r from-transparent via-indigo-200 to-transparent" />
                    ),
                    // Important terms highlighting
                    strong: ({ children }) => (
                      <strong className="font-semibold text-indigo-700 bg-indigo-50 px-1 rounded">
                        {children}
                      </strong>
                    ),
                    // Code blocks with syntax highlighting
                    code: ({ inline, className, children }: { inline?: boolean; className?: string; children: React.ReactNode }) => {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <div className="relative group">
                          <div className="absolute -inset-2 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-lg blur opacity-0 group-hover:opacity-100 transition-opacity" />
                          <pre className="relative !mt-0 !mb-0 bg-gradient-to-br from-gray-900 via-[#1e1b4b] to-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto font-mono text-sm shadow-lg border border-indigo-500/20">
                            <code className={className}>{children}</code>
                          </pre>
                        </div>
                      ) : (
                        <code className="px-1.5 py-0.5 bg-indigo-50 text-indigo-600 rounded font-mono text-sm">
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {localNotes}
                </ReactMarkdown>
              </div>
            );
          }
    
          if (isGenerating) {
            return (
              <div className="flex flex-col items-center justify-center h-full">
                <div className="w-24 h-24 mb-4">
                  <LoadingNotes className="w-full h-full" />
                </div>
                <p className="text-lg font-medium text-gray-900">Generating Notes...</p>
              </div>
            );
          }
    
          return (
            <div className="flex flex-col items-center justify-center h-full">
              <button
                onClick={handleGenerate}
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
                className="group relative w-32 h-32 perspective"
              >
                <div className={`absolute inset-0 transform transition-transform duration-500 preserve-3d ${isHovered ? 'rotate-y-180 scale-110' : ''}`}>
                  <div className="absolute inset-0 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center backface-hidden shadow-xl">
                    <PenLine className="w-16 h-16 text-white transform transition-transform group-hover:scale-110" />
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-br from-teal-600 to-emerald-500 rounded-xl flex items-center justify-center backface-hidden rotate-y-180 shadow-xl">
                    <span className="text-white font-bold">Generate</span>
                  </div>
                </div>
                <div className="absolute -inset-4 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity" />
              </button>
              <div className="mt-6 flex flex-col gap-4">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min="1"
                      placeholder="Start Page"
                      value={startPage || ''}
                      onChange={(e) => setStartPage(e.target.value ? Number(e.target.value) : undefined)}
                      className="w-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    />
                    <span className="text-gray-500">to</span>
                    <input
                      type="number"
                      min="1"
                      placeholder="End Page"
                      value={endPage || ''}
                      onChange={(e) => setEndPage(e.target.value ? Number(e.target.value) : undefined)}
                      className="w-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                  <span className="text-gray-500">OR</span>
                  <input
                    type="text"
                    placeholder="Specific pages (e.g., 1,3,5)"
                    value={specificPages}
                    onChange={(e) => setSpecificPages(e.target.value)}
                    className="w-64 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
                {error && (
                  <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-2">
                    {error}
                  </div>
                )}
              </div>
              <p className="mt-6 text-gray-600 text-center max-w-sm">
                Generate comprehensive study notes with key points and detailed explanations
              </p>
            </div>
          );

      case 'flashcards':
        if (localFlashcards?.length) {
          return (
            <div className="flex items-center justify-center h-full">
              <div className="w-full max-w-2xl">
                <div className="flex items-center justify-between gap-4">
                  <button
                    onClick={handlePrevCard}
                    className="p-2 hover:bg-gray-100 rounded-full transition-all duration-200 hover:scale-110"
                    title="Previous card"
                  >
                    <ChevronLeft className="h-6 w-6 text-gray-600" />
                  </button>
                  
                  <div className="flex-1">
                    <Flashcard
                      key={currentFlashcardIndex}
                      question={localFlashcards[currentFlashcardIndex].question}
                      answer={localFlashcards[currentFlashcardIndex].answer}
                      slideDirection={slideDirection}
                      onSwipe={(direction) => {
                        if (direction === 'left') {
                          handleNextCard();
                        } else {
                          handlePrevCard();
                        }
                      }}
                    />
                  </div>

                  <button
                    onClick={handleNextCard}
                    className="p-2 hover:bg-gray-100 rounded-full transition-all duration-200 hover:scale-110"
                    title="Next card"
                  >
                    <ChevronRight className="h-6 w-6 text-gray-600" />
                  </button>
                </div>
                
                <div className="text-center mt-4 text-sm text-gray-500 flex items-center justify-center gap-2">
                  Card {currentFlashcardIndex + 1} of {localFlashcards.length}
                  <button 
                    onClick={handleShuffle}
                    className="p-2 hover:bg-gray-100 rounded-full transition-all duration-200 hover:scale-110"
                    title="Shuffle cards"
                  >
                    <Shuffle className="h-6 w-6 text-indigo-600 animate-pulse" />
                  </button>
                </div>
              </div>
            </div>
          );
        }

        if (isGenerating) {
          return (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="w-24 h-24 mb-4">
                <LoadingBrain className="w-full h-full" />
              </div>
              <p className="text-lg font-medium text-gray-900">Generating Flashcards...</p>
            </div>
          );
        }

        return (
          <div className="flex flex-col items-center justify-center h-full">
            <button
              onClick={handleGenerate}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
              className="group relative w-32 h-32 perspective"
            >
              <div className={`absolute inset-0 transform transition-transform duration-500 preserve-3d ${isHovered ? 'rotate-y-180 scale-110' : ''}`}>
                <div className="absolute inset-0 bg-gradient-to-br from-pink-500 to-rose-600 rounded-xl flex items-center justify-center backface-hidden shadow-xl">
                  <Brain className="w-16 h-16 text-white transform transition-transform group-hover:scale-110" />
                </div>
                <div className="absolute inset-0 bg-gradient-to-br from-rose-600 to-pink-500 rounded-xl flex items-center justify-center backface-hidden rotate-y-180 shadow-xl">
                  <span className="text-white font-bold">Generate</span>
                </div>
              </div>
              <div className="absolute -inset-4 bg-gradient-to-r from-pink-500 to-rose-600 rounded-xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity" />
            </button>
            <div className="mt-6 flex flex-col gap-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="1"
                    placeholder="Start"
                    value={startPage || ''}
                    onChange={(e) => setStartPage(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                  <span className="text-gray-500">to</span>
                  <input
                    type="number"
                    min="1"
                    placeholder="End"
                    value={endPage || ''}
                    onChange={(e) => setEndPage(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
                <span className="text-gray-500">or</span>
                <input
                  type="text"
                  placeholder="Specific pages (e.g., 1,3,5)"
                  value={specificPages}
                  onChange={(e) => setSpecificPages(e.target.value)}
                  className="w-48 px-2 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
              {error && (
                <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-2">
                  {error}
                </div>
              )}
            </div>
            <p className="mt-4 text-gray-600 text-center max-w-sm">
              Generate AI-powered flashcards to help you memorize key concepts from your document
            </p>
          </div>
        );
  
  // Update the test case in renderTabContent
      case 'test':
        if (localTest.length > 0) {
          if (testSubmitted) {
            return (
              <div className="space-y-8">
                <div className="bg-gradient-to-br from-white to-purple-100 p-8 rounded-xl shadow-lg border border-purple-100">
                  <div className="text-center mb-8">
                    <div className="relative inline-block">
                      <div className={`text-6xl font-bold ${
                        testScore >= 70 ? 'text-green-600' : 
                        testScore >= 50 ? 'text-purple-600' : 
                        'text-red-600'
                      }`}>
                        {testScore}%
                      </div>
                      <div className="mt-2 text-gray-600">
                        {testScore >= 70 ? 'Excellent work!' : 
                         testScore >= 50 ? 'Good effort!' : 
                         'Keep practicing!'}
                      </div>
                    </div>
                  </div>
    
                  <div className="space-y-6">
                    {localTest.map((question, index) => {
                      const userAnswer = testAnswers.find(a => a.questionIndex === index)?.selectedAnswer;
                      const isCorrect = userAnswer === question.correctAnswer;
    
                      return (
                        <div 
                          key={index}
                          className={`p-6 rounded-lg border ${
                            isCorrect ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'
                          }`}
                        >
                          <div className="flex items-start space-x-4">
                            <span className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                              isCorrect ? 'bg-green-500' : 'bg-red-500'
                            } text-white`}>
                              {index + 1}
                            </span>
                            <div className="flex-1">
                              <h3 className="text-lg font-semibold text-gray-900">{question.question}</h3>
                              <div className="mt-4 space-y-2">
                                <div className={`text-sm ${isCorrect ? 'text-green-700' : 'text-red-700'}`}>
                                  Your answer: {userAnswer}
                                </div>
                                {!isCorrect && (
                                  <div className="text-sm text-green-700">
                                    Correct answer: {question.correctAnswer}
                                  </div>
                                )}
                                <div className="mt-2 text-sm text-gray-600">
                                  {question.explanation}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
    
                  <button
                    onClick={() => setTestSubmitted(false)}
                    className="mt-8 w-full px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-lg font-semibold hover:from-indigo-600 hover:to-purple-600 transition-all duration-200 transform hover:scale-[1.02]"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            );
          }
    
          return (
            <div className="space-y-8">
              {localTest.map((question, index) => (
                <div 
                  key={index} 
                  className="bg-gradient-to-br from-white to-purple-100 p-8 rounded-xl shadow-lg border border-purple-100 transform transition-all duration-300 hover:scale-[1.02] hover:shadow-xl"
                >
                  <div className="flex items-start space-x-4">
                    <span className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-500 text-white rounded-full flex items-center justify-center font-bold">
                      {index + 1}
                    </span>
                    <h3 className="text-xl font-semibold text-gray-900 flex-1">{question.question}</h3>
                  </div>
    
                  <div className="mt-6 pl-12">
                    {question.type === 'multiple_choice' ? (
                      <div className="space-y-3">
                        {question.options?.map((option, optionIndex) => (
                          <label 
                            key={optionIndex} 
                            className="flex items-center p-4 bg-white rounded-lg border border-purple-100 cursor-pointer transform transition-all duration-200 hover:scale-[1.01] hover:shadow-md group"
                          >
                            <input
                              type="radio"
                              name={`question-${index}`}
                              value={option}
                              checked={testAnswers.find(a => a.questionIndex === index)?.selectedAnswer === option}
                              onChange={() => handleAnswerSelect(index, option)}
                              className="w-4 h-4 text-purple-600 focus:ring-purple-500 focus:ring-offset-2"
                            />
                            <span className="ml-3 text-gray-800 group-hover:text-purple-700 transition-colors">
                              {option}
                            </span>
                            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/0 to-purple-500/0 group-hover:from-purple-500/5 group-hover:via-purple-500/5 group-hover:to-purple-500/0 rounded-lg transition-all duration-500" />
                          </label>
                        ))}
                      </div>
                    ) : (
                      <div className="flex space-x-6">
                        {['True', 'False'].map((option) => (
                          <label 
                            key={option}
                            className="flex-1 flex items-center p-4 bg-white rounded-lg border border-purple-100 cursor-pointer transform transition-all duration-200 hover:scale-[1.01] hover:shadow-md group"
                          >
                            <input
                              type="radio"
                              name={`question-${index}`}
                              value={option}
                              checked={testAnswers.find(a => a.questionIndex === index)?.selectedAnswer === option}
                              onChange={() => handleAnswerSelect(index, option)}
                              className="w-4 h-4 text-purple-600 focus:ring-purple-500 focus:ring-offset-2"
                            />
                            <span className="ml-3 text-gray-800 group-hover:text-purple-700 transition-colors">
                              {option}
                            </span>
                            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/0 to-purple-500/0 group-hover:from-purple-500/5 group-hover:via-purple-500/5 group-hover:to-purple-500/0 rounded-lg transition-all duration-500" />
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
    
              <div className="mt-8">
                <button
                  onClick={handleTestSubmit}
                  className="w-full px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-lg font-semibold hover:from-indigo-600 hover:to-purple-600 transition-all duration-200 transform hover:scale-[1.02] shadow-lg"
                >
                  Submit Test
                </button>
              </div>
            </div>
          );
        }
        if (isGenerating) {
          return (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="w-24 h-24 mb-4">
                <LoadingTest className="w-full h-full" />
              </div>
              <p className="text-lg font-medium text-gray-900">Generating Test Questions...</p>
            </div>
          );
        }
      
        return (
          <div className="flex flex-col items-center justify-center h-full">
            <button
              onClick={handleGenerate}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
              className="group relative w-32 h-32 perspective"
            >
              <div className={`absolute inset-0 transform transition-transform duration-500 preserve-3d ${isHovered ? 'rotate-y-180 scale-110' : ''}`}>
                <div className="absolute inset-0 bg-[conic-gradient(at_top_right,_var(--tw-gradient-stops))] from-amber-500 via-yellow-500 to-amber-500 rounded-xl flex items-center justify-center backface-hidden shadow-xl overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-amber-400/50 to-transparent animate-[spin_4s_linear_infinite]" />
                  <div className="absolute inset-0 bg-gradient-to-tl from-yellow-400/50 to-transparent animate-[spin_4s_linear_infinite_reverse]" />
                  <HelpCircle className="w-16 h-16 text-white transform transition-transform group-hover:scale-110 relative z-10" />
                </div>
                <div className="absolute inset-0 bg-[conic-gradient(at_top_right,_var(--tw-gradient-stops))] from-yellow-500 via-amber-500 to-yellow-500 rounded-xl flex items-center justify-center backface-hidden rotate-y-180 shadow-xl overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-yellow-400/50 to-transparent animate-[spin_4s_linear_infinite]" />
                  <div className="absolute inset-0 bg-gradient-to-tl from-amber-400/50 to-transparent animate-[spin_4s_linear_infinite_reverse]" />
                  <span className="text-white font-bold relative z-10">Generate</span>
                </div>
              </div>
              <div className="absolute -inset-4 bg-gradient-to-r from-amber-500 to-yellow-500 rounded-xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity" />
            </button>
            <div className="mt-6 flex flex-col gap-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="1"
                    placeholder="Start"
                    value={startPage || ''}
                    onChange={(e) => setStartPage(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                  <span className="text-gray-500">to</span>
                  <input
                    type="number"
                    min="1"
                    placeholder="End"
                    value={endPage || ''}
                    onChange={(e) => setEndPage(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
                <span className="text-gray-500">or</span>
                <input
                  type="text"
                  placeholder="Specific pages (e.g., 1,3,5)"
                  value={specificPages}
                  onChange={(e) => setSpecificPages(e.target.value)}
                  className="w-48 px-2 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
              {error && (
                <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-2">
                  {error}
                </div>
              )}
            </div>
            <p className="mt-4 text-gray-600 text-center max-w-sm">
              Generate an interactive test with multiple-choice and true/false questions to assess your understanding
            </p>
          </div>
        );

      case 'vocalize':
        return <VocalizeTab document={document} onUpdate={onDocumentsChange} />;

      
      default:
        return null;
    }
  };

  return (
    <div className={`
      w-full h-[92vh] md:h-[calc(100vh-4rem)] flex flex-col
    `}>
      <div className="flex-1 relative overflow-hidden">
        {error && (
          <div className="absolute top-4 right-4 bg-red-50 text-red-600 px-4 py-2 rounded-lg flex items-center gap-2 animate-slideUp">
            <AlertCircle className="h-4 w-4" />
            {error}
          </div>
        )}
        <PanelGroup direction={isMobile ? "vertical" : "horizontal"} className="h-full">
          {isPdfVisible && (
            <>
              <Panel defaultSize={isMobile && document.type === 'youtube' ? 35 : 50} minSize={isMobile && document.type === 'youtube' ? 30 : isMobile ? 40 : 30}>
                <div className={`h-full bg-gray-100 overflow-hidden relative`}>
                  {document.type === 'youtube' ? (
                    document.youtube_link ? (
                      <YoutubeViewer 
                        url={document.youtube_link} 
                        onPlayerReady={setYoutubePlayerRef}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-gray-500">YouTube URL not available</p>
                      </div>
                    )
                  ) : document.type === 'lecture' ? (
                    <LectureTranscript 
                      content={document.content}
                      documentId={document.id}
                      audioUrl={document.audioUrl}
                      onTranscriptUpdate={handleTranscriptUpdate}
                    />
                  ) : document.fileUrl ? (
                    <PdfViewer
                      url={document.fileUrl}
                      scale={scale}
                      onScaleChange={setScale}
                      onTextSelect={(text) => {
                        setSelectedText(text);
                        setSelectedImage(null);
                        setActiveTab('chat');
                      }}
                      onAreaSelect={(imageData) => {
                        setSelectedImage(imageData);
                        setSelectedText(null);
                        setActiveTab('chat');
                      }}
                      documentId={document.id}
                      onContentUpdate={handleContentUpdate}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full">
                      <p className="text-gray-500">No file available</p>
                    </div>
                  )}
                </div>
              </Panel>
              {!isMobile && (
                <PanelResizeHandle className="w-2 bg-gray-200 hover:bg-gray-300 transition-colors">
                  <div className="w-1 h-8 bg-gray-400 rounded-full mx-auto my-2" />
                </PanelResizeHandle>
              )}
              {isMobile && (
                <div className="h-[1px] bg-gray-200 w-full" />
              )}
            </>
          )}

          {/* Content Panel */}
          <Panel minSize={isMobile ? 50 : 30}>
            <div className="h-full flex flex-col bg-white">
              {/* Header */}
              <div className="flex-shrink-0 border-b border-gray-200 px-4 md:px-6 py-2 md:py-4">
                <div className="flex justify-between items-center mb-2 md:mb-4">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setIsPdfVisible(!isPdfVisible)}
                      className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                      title={isPdfVisible ? "Hide Document" : "Show Document"}
                    >
                      {isPdfVisible ? (
                        <PanelLeftClose className="h-5 w-5 text-gray-600" />
                      ) : (
                        <PanelLeftOpen className="h-5 w-5 text-gray-600" />
                      )}
                    </button>
                    {/* Hide title on mobile */}
                    <div className="hidden md:block">
                      {isEditingTitle ? (
                        <form 
                          onSubmit={(e) => {
                            e.preventDefault();
                            handleTitleUpdate();
                          }}
                          className="flex-1"
                        >
                          <input
                            ref={titleInputRef}
                            type="text"
                            value={newTitle}
                            onChange={(e) => setNewTitle(e.target.value)}
                            onBlur={handleTitleUpdate}
                            className="w-full px-2 py-1 text-xl md:text-2xl font-bold text-gray-900 border-b-2 border-indigo-500 focus:outline-none focus:border-indigo-600 bg-transparent"
                            placeholder="Enter document title..."
                          />
                        </form>
                      ) : (
                        <h1 
                          className="text-xl md:text-2xl font-bold text-gray-900 truncate cursor-pointer hover:text-indigo-600 transition-colors group flex items-center gap-2"
                          onClick={() => setIsEditingTitle(true)}
                          title="Click to rename"
                        >
                          {document.title}
                          <Pencil className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </h1>
                      )}
                    </div>
                  </div>
                  {document.fileUrl && (
                    <a
                      href={document.fileUrl}
                      download={document.title}
                      className="inline-flex items-center px-3 py-1.5 md:px-4 md:py-2 text-sm bg-indigo-600 text-white rounded-md hover:bg-indigo-500 transition-colors"
                    >
                      <FileDown className="h-4 w-4 mr-2" />
                      <span className="hidden md:inline">Download</span>
                    </a>
                  )}
                </div>

                {/* Tabs with scroll buttons */}
                <div className="relative -mb-[1px]">
                  <div 
                    ref={tabsContainerRef}
                    className="flex overflow-x-auto overflow-y-hidden whitespace-nowrap border-b border-gray-200 scrollbar-hide"
                  >
                    {tabs.map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`inline-flex items-center shrink-0 gap-1.5 px-4 py-2 text-sm font-medium transition-colors ${
                          activeTab === tab.id
                            ? 'text-indigo-600 bg-white border-t border-x border-gray-200'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                        style={{
                          marginBottom: '-1px',
                          borderBottom: activeTab === tab.id ? '1px solid white' : undefined
                        }}
                      >
                        <tab.icon className="h-4 w-4" />
                        <span>{tab.label}</span>
                        {tab.beta && (
                          <span className="text-[10px] font-medium px-1.5 py-0.5 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 text-indigo-600 rounded-full border border-indigo-200/20">
                            BETA
                          </span>
                        )} 
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Tab Content */}
              <div className={`flex-1 min-h-0 overflow-auto p-4 md:p-6 ${
                !isMobile && activeTab === 'notes' ? 'max-h-[calc(100vh-8rem)]' : ''
              }`}>
                <div className={`${isMobile && activeTab === 'notes' ? 'max-w-[100vw] break-words overflow-x-hidden [&_table]:overflow-x-auto [&_table]:max-w-full [&_table]:block [&_table]:w-fit' : ''}`}>
                  {renderTabContent()}
                </div>
              </div>
            </div>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}
