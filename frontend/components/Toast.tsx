type ToastProps = {
  message: string;
  visible: boolean;
};

export default function Toast({ message, visible }: ToastProps) {
  if (!visible) return null;
  return (
    <div className="fixed bottom-6 right-6 border border-hairline px-4 py-2 text-xs uppercase tracking-[0.2em] bg-black text-white">
      {message}
    </div>
  );
}
