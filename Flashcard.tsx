import React, { useState, useRef, useEffect } from 'react';

interface FlashcardProps {
  question: string;
  answer: string;
  onSwipe?: (direction: 'left' | 'right') => void;
  slideDirection?: 'left' | 'right';
}

export default function Flashcard({ question, answer, onSwipe, slideDirection }: FlashcardProps) {
  const [isFlipped, setIsFlipped] = useState(false);
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);
  const [questionFontSize, setQuestionFontSize] = useState(20);
  const [answerFontSize, setAnswerFontSize] = useState(20);
  const cardRef = useRef<HTMLDivElement>(null);
  const questionTextRef = useRef<HTMLDivElement>(null);
  const answerTextRef = useRef<HTMLDivElement>(null);

  // Reset flip state when card changes
  useEffect(() => {
    setIsFlipped(false);
  }, [question, answer]);

  useEffect(() => {
    const adjustFontSize = (element: HTMLDivElement, setText: (size: number) => void) => {
      let fontSize = 20;
      element.style.fontSize = `${fontSize}px`;
      
      while (element.scrollHeight > element.clientHeight && fontSize > 12) {
        fontSize--;
        element.style.fontSize = `${fontSize}px`;
      }
      setText(fontSize);
    };

    if (questionTextRef.current) {
      adjustFontSize(questionTextRef.current, setQuestionFontSize);
    }
    if (answerTextRef.current) {
      adjustFontSize(answerTextRef.current, setAnswerFontSize);
    }
  }, [question, answer]);

  const handleTouchStart = (e: React.TouchEvent) => {
    setTouchStart(e.targetTouches[0].clientX);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const handleTouchEnd = () => {
    if (!touchStart || !touchEnd || !onSwipe) return;
    
    const distance = touchStart - touchEnd;
    const isSignificantSwipe = Math.abs(distance) > 50;
    
    if (isSignificantSwipe) {
      if (distance > 0) {
        onSwipe('left');
      } else {
        onSwipe('right');
      }
    }
    
    setTouchStart(null);
    setTouchEnd(null);
  };

  const slideAnimation = slideDirection === 'left' ? 'animate-slideLeft' : 
                        slideDirection === 'right' ? 'animate-slideRight' : '';

  return (
    <div
      ref={cardRef}
      className={`relative h-64 w-full md:w-[32rem] mx-auto cursor-pointer perspective touch-pan-y ${slideAnimation}`}
      onClick={() => setIsFlipped(!isFlipped)}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      <div
        className={`absolute inset-0 w-full h-full duration-700 preserve-3d ${
          isFlipped ? 'rotate-y-180' : ''
        }`}
      >
        {/* Question Side */}
        <div className="absolute inset-0 w-full h-full backface-hidden bg-gradient-to-br from-white to-indigo-50 border-2 border-indigo-100 rounded-xl p-6 flex flex-col items-center justify-center text-center shadow-lg transform transition-all duration-300">
          <p className="text-sm font-medium text-indigo-600 mb-4">Question</p>
          <div ref={questionTextRef} className="h-[120px] flex items-center justify-center">
            <p className="text-gray-900" style={{ fontSize: `${questionFontSize}px` }}>{question}</p>
          </div>
          <p className="mt-4 text-sm text-gray-500">Click to reveal answer</p>
        </div>

        {/* Answer Side */}
        <div className="absolute inset-0 w-full h-full backface-hidden rotate-y-180 bg-gradient-to-br from-indigo-50 to-white border-2 border-indigo-100 rounded-xl p-6 flex flex-col items-center justify-center text-center shadow-lg transform transition-transform duration-700">
          <p className="text-sm font-medium text-indigo-600 mb-4">Answer</p>
          <div ref={answerTextRef} className="h-[120px] flex items-center justify-center">
            <p className="text-gray-900" style={{ fontSize: `${answerFontSize}px` }}>{answer}</p>
          </div>
          <p className="mt-4 text-sm text-gray-500">Click to see question</p>
        </div>
      </div>
    </div>
  );
}