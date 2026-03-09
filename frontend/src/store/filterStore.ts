import { create } from 'zustand';

export interface FilterState {
  source: string;
  category: string;
  days: number;
  search: string;
  ordering: string;
  mode: 'trending' | 'emerging';
}

interface FilterStore extends FilterState {
  setSource: (source: string) => void;
  setCategory: (category: string) => void;
  setDays: (days: number) => void;
  setSearch: (search: string) => void;
  setOrdering: (ordering: string) => void;
  setMode: (mode: 'trending' | 'emerging') => void;
  setFilters: (filters: Partial<FilterState>) => void;
  resetFilters: () => void;
}

const defaultFilters: FilterState = {
  source: 'all',
  category: 'all',
  days: 7,
  search: '',
  ordering: '-momentum_score',
  mode: 'trending',
};

export const useFilterStore = create<FilterStore>((set) => ({
  ...defaultFilters,

  setSource: (source) => set({ source }),
  setCategory: (category) => set({ category }),
  setDays: (days) => set({ days }),
  setSearch: (search) => set({ search }),
  setOrdering: (ordering) => set({ ordering }),
  setMode: (mode) => set({ mode }),

  setFilters: (filters) => set((state) => ({ ...state, ...filters })),

  resetFilters: () => set(defaultFilters),
}));